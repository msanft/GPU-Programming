#include <cmath>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define TILE_WIDTH 32
#define MATRIX_SIZE 5000 // square
#define EPSILON 0.1f
#define STREAM_COUNT 2

// <--- HELPER FUNCTIONS --->

/*
populate_matrix fills the matrix `res` with random float values.

Values are in the range [0, 25.5] (25.5 due to 0xff=255/10) to avoid precision
issues in comparison.

See https://docs.nvidia.com/cuda/floating-point/index.html.

The caller is expected to seed the RNG.
*/
void populate_matrix(float_t *res, uint32_t width, uint32_t height) {
  for (uint32_t i = 0; i < height; i++)
    for (uint32_t j = 0; j < width; j++)
      // avoid precision issues in comparison by giving an upper bound
      res[i * width + j] = (float_t)(rand() & 0xFF) / 10.0f;
}

/*
get_time_diff_ns calculates the difference between two timespec structs in
nanoseconds.
*/
int64_t get_time_diff_ns(struct timespec start, struct timespec end) {
  int64_t diff = (int64_t)(end.tv_sec - start.tv_sec) * (int64_t)1e9 +
                 (int64_t)(end.tv_nsec - start.tv_nsec);
  return diff;
}

// <--- CPU FUNCTIONS --->

/*
_multiply_matrices_cpu performs matrix multiplication on `m1` and `m2` and
stores the result in `res`.

It should not be used directly; instead, use the wrapper function
`multiply_matrices_cpu`.
*/
void _multiply_matrices_cpu(float_t *res, float_t *m1, float_t *m2,
                            uint32_t result_width, uint32_t result_height,
                            uint32_t shared_dim) {
  for (uint32_t i = 0; i < result_height; i++) {
    for (uint32_t j = 0; j < result_width; j++) {
      float_t p_value = 0;

      for (uint32_t k = 0; k < shared_dim; k++)
        p_value += m1[i * shared_dim + k] * m2[k * result_width + j];

      res[i * result_width + j] = p_value;
    }
  }
}

/*
multiply_matrices_cpu performs matrix multiplication on `m1` and `m2` using the
local tiled matrix multiplication algorithm and stores the result in `res`.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t multiply_matrices_cpu(float_t *res, float_t *m1, float_t *m2,
                              uint32_t result_width, uint32_t result_height,
                              uint32_t shared_dim) {
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _multiply_matrices_cpu(res, m1, m2, result_width, result_height, shared_dim);

  clock_gettime(CLOCK_MONOTONIC, &end);

  return get_time_diff_ns(start, end);
}

/*
compare_outputs checks if the matrices in `res_1` and `res_2`
match by comparing each element with the `epsilon` error margin.

See https://docs.nvidia.com/cuda/floating-point/index.html.

If a mismatch is found, it will be printed to stderr and 1 will be returned.
Otherwise, 0 will be returned.
*/
uint32_t compare_outputs(float_t *res_1, float_t *res_2, float_t epsilon,
                         uint32_t width, uint32_t height) {
  for (uint32_t y = 0; y < height; y++)
    for (uint32_t x = 0; x < width; x++)
      if ((fabs(res_1[y * width + x] - res_2[y * width + x]) /
           fabs(res_1[y * width + x])) > epsilon) {
        fprintf(stderr, "Results differ significantly at index (%d, %d).\n", x,
                y);
        fprintf(stderr, "Result 1: %f, Result 2: %f\n", res_1[y * width + x],
                res_2[y * width + x]);
        fprintf(stderr, "Relative difference: %f\n",
                fabs(res_1[y * width + x] - res_2[y * width + x]) /
                    fabs(res_1[y * width + x]));
        return 1;
      }

  return 0;
}

// <--- GPU FUNCTIONS --->

/*
_multiply_matrices_gpu performs matrix multiplication on `m1` and `m2` using the
local tiled matrix multiplication algorithm and stores the result in `res`.

It should not be used directly; instead, use the wrapper function
`multiply_matrices_gpu`.
*/
__global__ void _multiply_matrices_gpu(float_t *res, float_t *m1, float_t *m2,
                                       uint32_t result_width,
                                       uint32_t result_height,
                                       uint32_t shared_dim) {
  uint32_t row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  uint32_t col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float_t p_value = 0;

  for (int k = 0; k < shared_dim; k++)
    p_value += m1[row * shared_dim + k] * m2[k * result_width + col];

  // bounds check required due to round-up in grid size
  if (row < result_height && col < result_width)
    res[row * result_width + col] = p_value;
}

/*
multiply_matrices_gpu_multistream performs matrix multiplication on `m1` and
`m2` and stores the result in `res`.

It uses multiple streams.

It also performs CUDA memory management and kernel invocation, using the
specified `block_width` and `block_height` for the kernel, as well as timing the
operation.

It returns the time taken to nanosecond-precision.
*/
int64_t multiply_matrices_gpu_multistream(float_t *res, float_t *m1,
                                          float_t *m2) {
  // Create compartments
  float_t *h_half_1 =
      (float_t *)malloc(MATRIX_SIZE * (MATRIX_SIZE / 2) * sizeof(float_t));
  float_t *h_half_2 =
      (float_t *)malloc(MATRIX_SIZE * (MATRIX_SIZE / 2) * sizeof(float_t));
  float_t *v_half_1 =
      (float_t *)malloc(MATRIX_SIZE * (MATRIX_SIZE / 2) * sizeof(float_t));
  float_t *v_half_2 =
      (float_t *)malloc(MATRIX_SIZE * (MATRIX_SIZE / 2) * sizeof(float_t));

  for (uint32_t i = 0; i < (MATRIX_SIZE / 2); i++) {
    memcpy(h_half_1 + (i * MATRIX_SIZE), m1 + (i * MATRIX_SIZE),
           MATRIX_SIZE * sizeof(float_t));
    memcpy(h_half_2 + (i * MATRIX_SIZE),
           m1 + ((MATRIX_SIZE / 2) * MATRIX_SIZE) + (i * MATRIX_SIZE),
           MATRIX_SIZE * sizeof(float_t));
  }

  for (uint32_t i = 0; i < (MATRIX_SIZE / 2); i++) {
    memcpy(v_half_1 + (i * (MATRIX_SIZE / 2)), m2 + (i * MATRIX_SIZE),
           (MATRIX_SIZE / 2) * sizeof(float_t));
    memcpy(v_half_2 + (i * (MATRIX_SIZE / 2)),
           m2 + (MATRIX_SIZE / 2) + (i * MATRIX_SIZE),
           (MATRIX_SIZE / 2) * sizeof(float_t));
  }

  float_t *upper_left = (float_t *)malloc((MATRIX_SIZE / 2) *
                                          (MATRIX_SIZE / 2) * sizeof(float_t));
  float_t *upper_right = (float_t *)malloc((MATRIX_SIZE / 2) *
                                           (MATRIX_SIZE / 2) * sizeof(float_t));
  float_t *lower_left = (float_t *)malloc((MATRIX_SIZE / 2) *
                                          (MATRIX_SIZE / 2) * sizeof(float_t));
  float_t *lower_right = (float_t *)malloc((MATRIX_SIZE / 2) *
                                           (MATRIX_SIZE / 2) * sizeof(float_t));

  float_t *d_h_half_1, *d_h_half_2, *d_v_half_1, *d_v_half_2, *d_upper_left,
      *d_upper_right, *d_lower_left, *d_lower_right;

  checkCudaErrors(cudaMalloc(&d_h_half_1, (MATRIX_SIZE * MATRIX_SIZE) / 2 *
                                              sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_h_half_2, (MATRIX_SIZE * MATRIX_SIZE) / 2 *
                                              sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_v_half_1, (MATRIX_SIZE * MATRIX_SIZE) / 2 *
                                              sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_v_half_2, (MATRIX_SIZE * MATRIX_SIZE) / 2 *
                                              sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_upper_left, (MATRIX_SIZE * MATRIX_SIZE) / 4 *
                                                sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_upper_right, (MATRIX_SIZE * MATRIX_SIZE) / 4 *
                                                 sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_lower_left, (MATRIX_SIZE * MATRIX_SIZE) / 4 *
                                                sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_lower_right, (MATRIX_SIZE * MATRIX_SIZE) / 4 *
                                                 sizeof(float_t)));

  dim3 grid(ceil(MATRIX_SIZE / float_t(TILE_WIDTH)),
            ceil(MATRIX_SIZE / float_t(TILE_WIDTH)));
  dim3 block(TILE_WIDTH, TILE_WIDTH);

  cudaStream_t *streams =
      (cudaStream_t *)malloc(STREAM_COUNT * sizeof(cudaStream_t));
  for (unsigned int i = 0; i < STREAM_COUNT; i++) {
    cudaStreamCreate(&streams[i]);
  }

  checkCudaErrors(cudaMemcpyAsync(
      d_h_half_1, h_half_1, (MATRIX_SIZE * MATRIX_SIZE) / 2 * sizeof(float_t),
      cudaMemcpyHostToDevice, streams[0]));
  checkCudaErrors(cudaMemcpyAsync(
      d_v_half_1, v_half_1, (MATRIX_SIZE * MATRIX_SIZE) / 2 * sizeof(float_t),
      cudaMemcpyHostToDevice, streams[0]));
  checkCudaErrors(cudaMemcpyAsync(
      d_h_half_2, h_half_2, (MATRIX_SIZE * MATRIX_SIZE) / 2 * sizeof(float_t),
      cudaMemcpyHostToDevice, streams[1]));
  checkCudaErrors(cudaMemcpyAsync(
      d_v_half_2, v_half_2, (MATRIX_SIZE * MATRIX_SIZE) / 2 * sizeof(float_t),
      cudaMemcpyHostToDevice, streams[1]));

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _multiply_matrices_gpu<<<grid, block>>>(d_upper_left, d_h_half_1, d_v_half_1,
                                          MATRIX_SIZE / 2, MATRIX_SIZE / 2,
                                          MATRIX_SIZE);
  _multiply_matrices_gpu<<<grid, block>>>(d_lower_right, d_h_half_2, d_v_half_2,
                                          MATRIX_SIZE / 2, MATRIX_SIZE / 2,
                                          MATRIX_SIZE);
  _multiply_matrices_gpu<<<grid, block>>>(d_upper_right, d_h_half_1, d_v_half_2,
                                          MATRIX_SIZE / 2, MATRIX_SIZE / 2,
                                          MATRIX_SIZE);
  _multiply_matrices_gpu<<<grid, block>>>(d_lower_left, d_h_half_2, d_v_half_1,
                                          MATRIX_SIZE / 2, MATRIX_SIZE / 2,
                                          MATRIX_SIZE);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(
      cudaMemcpyAsync(upper_left, d_upper_left,
                      (MATRIX_SIZE * MATRIX_SIZE) / 4 * sizeof(float_t),
                      cudaMemcpyDeviceToHost, streams[0]));
  checkCudaErrors(
      cudaMemcpyAsync(lower_right, d_lower_right,
                      (MATRIX_SIZE * MATRIX_SIZE) / 4 * sizeof(float_t),
                      cudaMemcpyDeviceToHost, streams[0]));

  checkCudaErrors(
      cudaMemcpyAsync(upper_right, d_upper_right,
                      (MATRIX_SIZE * MATRIX_SIZE) / 4 * sizeof(float_t),
                      cudaMemcpyDeviceToHost, streams[0]));
  checkCudaErrors(
      cudaMemcpyAsync(lower_left, d_lower_left,
                      (MATRIX_SIZE * MATRIX_SIZE) / 4 * sizeof(float_t),
                      cudaMemcpyDeviceToHost, streams[0]));

  for (uint32_t i = 0; i < (MATRIX_SIZE / 2); i++) {
    memcpy(res + (i * MATRIX_SIZE), upper_left + (i * (MATRIX_SIZE / 2)),
           (MATRIX_SIZE / 2) * sizeof(float_t));
    memcpy(res + (MATRIX_SIZE / 2) + (i * MATRIX_SIZE),
           upper_right + (i * (MATRIX_SIZE / 2)),
           (MATRIX_SIZE / 2) * sizeof(float_t));
    memcpy(res + (MATRIX_SIZE * (MATRIX_SIZE / 2)) + (i * MATRIX_SIZE),
           lower_left + (i * (MATRIX_SIZE / 2)),
           (MATRIX_SIZE / 2) * sizeof(float_t));
    memcpy(res + (MATRIX_SIZE * (MATRIX_SIZE / 2)) + (MATRIX_SIZE / 2) +
               (i * MATRIX_SIZE),
           lower_right + (i * (MATRIX_SIZE / 2)),
           (MATRIX_SIZE / 2) * sizeof(float_t));
  }

  checkCudaErrors(cudaFree(d_h_half_1));
  checkCudaErrors(cudaFree(d_h_half_2));
  checkCudaErrors(cudaFree(d_v_half_1));
  checkCudaErrors(cudaFree(d_v_half_2));
  checkCudaErrors(cudaFree(d_upper_left));
  checkCudaErrors(cudaFree(d_upper_right));
  checkCudaErrors(cudaFree(d_lower_left));
  checkCudaErrors(cudaFree(d_lower_right));

  free(h_half_1);
  free(h_half_2);
  free(v_half_1);
  free(v_half_2);
  free(upper_left);
  free(upper_right);
  free(lower_left);
  free(lower_right);

  return get_time_diff_ns(start, end);
}

/*
multiply_matrices_gpu performs matrix multiplication on `m1` and `m2` using
the local tiled matrix multiplication algorithm and stores the result in
`res`.

It does not use multiple streams.

It also performs CUDA memory management and kernel invocation, using the
specified `block_width` and `block_height` for the kernel, as well as timing
the operation.

It returns the time taken to nanosecond-precision.
*/
int64_t multiply_matrices_gpu(float_t *res, float_t *m1, float_t *m2,
                              uint32_t result_width, uint32_t result_height,
                              uint32_t shared_dim) {
  float_t *d_res, *d_m1, *d_m2;

  checkCudaErrors(
      cudaMalloc(&d_res, result_width * result_height * sizeof(float_t)));
  checkCudaErrors(
      cudaMalloc(&d_m1, shared_dim * result_height * sizeof(float_t)));
  checkCudaErrors(
      cudaMalloc(&d_m2, result_width * shared_dim * sizeof(float_t)));

  checkCudaErrors(cudaMemcpy(d_m1, m1,
                             shared_dim * result_height * sizeof(float_t),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_m2, m2,
                             result_width * shared_dim * sizeof(float_t),
                             cudaMemcpyHostToDevice));

  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid((result_width + TILE_WIDTH - 1) / TILE_WIDTH,
            (result_height + TILE_WIDTH - 1) / TILE_WIDTH);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _multiply_matrices_gpu<<<grid, block>>>(d_res, d_m1, d_m2, result_width,
                                          result_height, shared_dim);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(cudaMemcpy(res, d_res,
                             result_width * result_height * sizeof(float_t),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_res));
  checkCudaErrors(cudaFree(d_m1));
  checkCudaErrors(cudaFree(d_m2));

  return get_time_diff_ns(start, end);
}

// <--- MAIN FUNCTION --->

int main(void) {
  // Seed the system RNG
  srand(time(NULL));

  uint32_t m_size[2] = {MATRIX_SIZE, MATRIX_SIZE};
  uint32_t n_size[2] = {MATRIX_SIZE, MATRIX_SIZE};
  uint32_t result_size[2] = {m_size[0], n_size[1]};

  int64_t cpu_time, gpu_time, gpu_time_multistream;

  float_t *m = (float_t *)malloc(m_size[0] * m_size[1] * sizeof(float_t));
  float_t *n = (float_t *)malloc(n_size[0] * n_size[1] * sizeof(float_t));
  float_t *p_cpu =
      (float_t *)malloc(result_size[0] * result_size[1] * sizeof(float_t));
  float_t *p_gpu =
      (float_t *)malloc(result_size[0] * result_size[1] * sizeof(float_t));
  float_t *p_gpu_multistream =
      (float_t *)malloc(result_size[0] * result_size[1] * sizeof(float_t));

  if (!m || !n || !p_cpu || !p_gpu || !p_gpu_multistream) {
    fprintf(stderr, "Failed to allocate memory.\n");
    goto ERR;
  }

  printf("Tile width: %d\n", TILE_WIDTH);

  populate_matrix(m, m_size[0], m_size[1]);
  populate_matrix(n, n_size[0], n_size[1]);

  printf("Timings:\n");

  cpu_time = multiply_matrices_cpu(p_cpu, m, n, result_size[0], result_size[1],
                                   m_size[1]);

  printf("\tCPU time: %ld ns\n", cpu_time);

  gpu_time = multiply_matrices_gpu(p_gpu, n, m, result_size[0], result_size[1],
                                   m_size[1]);

  printf("\tGPU time: %ld ns\n", gpu_time);

  if (!compare_outputs(p_cpu, p_gpu, EPSILON, result_size[0], result_size[1]))
    goto ERR;

  gpu_time_multistream =
      multiply_matrices_gpu_multistream(p_gpu_multistream, m, n);

  printf("\tGPU time (multistream): %ld ns\n", gpu_time_multistream);

  if (!compare_outputs(p_gpu, p_gpu_multistream, EPSILON, result_size[0],
                       result_size[1]))
    goto ERR;

ERR:
  free(m);
  free(n);
  free(p_cpu);
  free(p_gpu);
  free(p_gpu_multistream);

  return 0;
}
