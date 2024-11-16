#include <cmath>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

enum Approach {
  ROW_TO_ROW_COPY,
  naive,
  SHARED_MEM,
};

// <--- HELPER FUNCTIONS --->

/*
populate_matrix fills the matrix `res` with random float values.

Values are in the range [0, 25.5] (25.5 due to 0xff=255/10) to avoid precision
issues in comparison.

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

/*
effectiveBandwidth calculates the effective bandwidth of a kernel in GB/s.
*/
int64_t effective_bandwidth(int64_t nsec, uint32_t width, uint32_t height,
                            uint32_t nreps) {
  int64_t total_bytes = (int64_t)width * height * (2 * sizeof(float)) * nreps;

  double_t sec_factor = nsec / 1e9;

  return (int64_t)((double_t)total_bytes / sec_factor / 1e9);
}

// <--- CPU FUNCTIONS --->

/*
_transpose_matrix_cpu performs matrix transposition on `m1` and
stores the result in `res`.

It should not be used directly; instead, use the wrapper function
`transpose_matrix_cpu`.
*/
void _transpose_matrix_cpu(float_t *res, float_t *m1, uint32_t height,
                           uint32_t width) {
  for (uint32_t i = 0; i < height; i++)
    for (uint32_t j = 0; j < width; j++)
      res[j * height + i] = m1[i * width + j];
}

/*
transpose_matrix_cpu performs matrix transposition on `m1` and stores
the result in `res`.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t transpose_matrix_cpu(float_t *res, float_t *m1, uint32_t height,
                             uint32_t width) {
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _transpose_matrix_cpu(res, m1, height, width);

  clock_gettime(CLOCK_MONOTONIC, &end);

  return get_time_diff_ns(start, end);
}

/*
compare_outputs checks if the matrices in `cpu_res` and `gpu_res`
match.

If a mismatch is found, it will be printed to stderr and 1 will be returned.
Otherwise, 0 will be returned.
*/
uint32_t compare_outputs(float_t *cpu_res, float_t *gpu_res, uint32_t width,
                         uint32_t height) {
  for (uint32_t i = 0; i < height; i++)
    for (uint32_t j = 0; j < width; j++)
      if (cpu_res[i * width + j] != gpu_res[i * width + j]) {
        fprintf(stderr, "CPU and GPU results do not match at index (%d, %d).\n",
                i, j);
        fprintf(stderr, "CPU: %f, GPU: %f\n", cpu_res[i * width + j],
                gpu_res[i * width + j]);
        return 1;
      }

  return 0;
}

// <--- GPU FUNCTIONS --->

/*
_transpose_matrix_gpu_rtr performs matrix transposition on `m1` using a
row-to-row-copy approach and stores the result in `res`.

It should not be used directly; instead, use the wrapper function
`transpose_matrix_gpu`.
*/
__global__ void _transpose_matrix_gpu_rtr(float_t *res, float_t *m1,
                                          uint32_t height, uint32_t width,
                                          uint32_t nreps) {
  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uint32_t i = x + width * y;

  for (uint32_t r = 0; r < nreps; r++)
    for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      if (y + j >= height)
        continue;
      res[i + j * width] = m1[i + j * width];
    }
}

/*
_transpose_matrix_gpu_naive performs matrix transposition on `m1` using a
naive approach and stores the result in `res`.

It should not be used directly; instead, use the wrapper function
`transpose_matrix_gpu`.
*/
__global__ void _transpose_matrix_gpu_naive(float_t *res, float_t *m1,
                                             uint32_t height, uint32_t width,
                                             uint32_t nreps) {
  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uint32_t i_1 = x + width * y;
  uint32_t i_2 = y + height * x;

  for (uint32_t r = 0; r < nreps; r++)
    for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      if (y + j >= height)
        continue;
      res[i_2 + j] = m1[i_1 + j * width];
    }
}

/*
_transpose_matrix_gpu_sharedmem performs matrix transposition on `m1` using the
shared memory and stores the result in `res`.

It should not be used directly; instead, use the wrapper function
`transpose_matrix_gpu`.
*/
__global__ void _transpose_matrix_gpu_sharedmem(float_t *res, float_t *m1,
                                                uint32_t height, uint32_t width,
                                                uint32_t nreps) {
  __shared__ float m1_shared[TILE_DIM][TILE_DIM + 1];

  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;
  uint32_t i = x + width * y;

  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x >= width || y + j >= height) {
      continue;
    }
    m1_shared[threadIdx.y + j][threadIdx.x] = m1[i + j * width];
  }

  __syncthreads();

  uint32_t x_out = blockIdx.y * TILE_DIM + threadIdx.x;
  uint32_t y_out = blockIdx.x * TILE_DIM + threadIdx.y;
  uint32_t i_out = x_out + height * y_out;

  for (unsigned int r = 0; r < nreps; r++)
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
      if (x_out >= height || y_out + j >= width)
        continue;
      res[i_out + j * height] = m1_shared[threadIdx.x][threadIdx.y + j];
    }
}

/*
transpose_matrix_gpu performs matrix transposition on `m1` using the given
`approach` and stores the result in `res`..

It also performs CUDA memory management and kernel invocation, using the
specified `block_width` and `block_height` for the kernel, as well as timing
the operation.

It returns the time taken to nanosecond-precision.
*/
int64_t transpose_matrix_gpu(float_t *res, float_t *m1, uint32_t height,
                             uint32_t width, uint32_t nreps,
                             Approach approach) {
  float_t *d_res, *d_m1;

  checkCudaErrors(cudaMalloc(&d_res, height * width * sizeof(float_t)));

  checkCudaErrors(cudaMalloc(&d_m1, height * width * sizeof(float_t)));

  checkCudaErrors(cudaMemcpy(d_m1, m1, height * width * sizeof(float_t),
                             cudaMemcpyHostToDevice));

  dim3 grid(ceil(width / float(TILE_DIM)), ceil(height / float(TILE_DIM)));
  dim3 block(TILE_DIM, BLOCK_ROWS);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  switch (approach) {
  case ROW_TO_ROW_COPY:
    _transpose_matrix_gpu_rtr<<<grid, block>>>(d_res, d_m1, height, width,
                                               nreps);
    break;
  case naive:
    _transpose_matrix_gpu_naive<<<grid, block>>>(d_res, d_m1, height, width,
                                                  nreps);
    break;
  case SHARED_MEM:
    _transpose_matrix_gpu_sharedmem<<<grid, block>>>(d_res, d_m1, height, width,
                                                     nreps);
    break;
  }

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(cudaMemcpy(res, d_res, height * width * sizeof(float_t),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_res));

  checkCudaErrors(cudaFree(d_m1));

  return get_time_diff_ns(start, end);
}

// <--- MAIN FUNCTION --->

int main(void) {
  // Seed the system RNG
  srand(time(NULL));

  uint32_t size[2] = {10000, 5000};

  uint32_t nreps = 1;

  int64_t cpu_time, gpu_time_rtr, gpu_time_naive, gpu_time_sharedmem;

  float_t *m1 = (float_t *)malloc(10000 * 5000 * sizeof(float_t));
  float_t *cpu_res = (float_t *)malloc(10000 * 5000 * sizeof(float_t));
  float_t *gpu_res_rtr = (float_t *)malloc(10000 * 5000 * sizeof(float_t));
  float_t *gpu_res_naive = (float_t *)malloc(10000 * 5000 * sizeof(float_t));
  float_t *gpu_res_sharedmem =
      (float_t *)malloc(10000 * 5000 * sizeof(float_t));

  populate_matrix(m1, size[0], size[1]);

  cpu_time = transpose_matrix_cpu(cpu_res, m1, size[0], size[1]);

  printf("CPU time: %ld ns\n", cpu_time);
  printf("Effective bandwidth: %ld GB/s\n",
         effective_bandwidth(cpu_time, size[0], size[1], 1));

  gpu_time_rtr = transpose_matrix_gpu(gpu_res_rtr, m1, size[0], size[1], nreps,
                                      ROW_TO_ROW_COPY);

  printf("GPU time (row-to-row copy): %ld ns\n", gpu_time_rtr);
  printf("Effective bandwidth (row-to-row copy): %ld GB/s\n",
         effective_bandwidth(gpu_time_rtr, size[0], size[1], nreps));

  // for row-to-row-copy, we check against equality to the input matrix
  if (compare_outputs(m1, gpu_res_rtr, size[0], size[1]))
    goto error;

  gpu_time_naive =
      transpose_matrix_gpu(gpu_res_naive, m1, size[0], size[1], nreps, naive);

  printf("GPU time (naive): %ld ns\n", gpu_time_naive);
  printf("Effective bandwidth (naive): %ld GB/s\n",
         effective_bandwidth(gpu_time_naive, size[0], size[1], nreps));

  if (compare_outputs(cpu_res, gpu_res_naive, size[0], size[1]))
    goto error;

  gpu_time_sharedmem = transpose_matrix_gpu(gpu_res_sharedmem, m1, size[0],
                                            size[1], nreps, SHARED_MEM);

  printf("GPU time (shared memory): %ld ns\n", gpu_time_sharedmem);
  printf("Effective bandwidth (shared memory): %ld GB/s\n",
         effective_bandwidth(gpu_time_sharedmem, size[0], size[1], nreps));

  if (compare_outputs(cpu_res, gpu_res_sharedmem, size[0], size[1]))
    goto error;

  free(m1);
  free(cpu_res);
  free(gpu_res_rtr);
  free(gpu_res_naive);
  free(gpu_res_sharedmem);

  return 0;

error:
  free(m1);
  free(cpu_res);
  free(gpu_res_rtr);
  free(gpu_res_naive);
  free(gpu_res_sharedmem);

  return 1;
}
