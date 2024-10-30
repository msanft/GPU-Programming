#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

// <--- HELPER FUNCTIONS --->

/*
populate_matrix fills the matrix `res` with random float values.

The caller is expected to seed the RNG.
*/
void populate_matrix(float_t *res, uint32_t rows, uint32_t cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      res[i * cols + j] = (float)rand();
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
_add_matrices_cpu performs matrix addition on `m1` and `m2` and stores the
result in `res`.

It should not be used directly; instead, use the wrapper function
`add_matrices_cpu`.
*/
void _add_matrices_cpu(float_t *res, float_t *m1, float_t *m2, uint32_t rows,
                       uint32_t cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      res[i * cols + j] = m1[i * cols + j] + m2[i * cols + j];
}

/*
add_matrices_cpu performs matrix addition on `m1` and `m2` and stores the result
in `res`.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t add_matrices_cpu(float_t *res, float_t *m1, float_t *m2, uint32_t rows,
                         uint32_t cols) {
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _add_matrices_cpu(res, m1, m2, rows, cols);

  clock_gettime(CLOCK_MONOTONIC, &end);

  return get_time_diff_ns(start, end);
}

/*
compare_outputs checks if the matrix addition results in `cpu_res` and `gpu_res`
match.

If a mismatch is found, it will be printed to stderr and 1 will be returned.
Otherwise, 0 will be returned.
*/
uint32_t compare_outputs(float_t *cpu_res, float_t *gpu_res, uint32_t rows,
                         uint32_t cols) {
  for (uint32_t i = 0; i < rows; i++)
    for (uint32_t j = 0; j < cols; j++)
      if (cpu_res[i * cols + j] != gpu_res[i * cols + j]) {
        fprintf(stderr, "CPU and GPU results do not match at index (%d, %d).\n",
                i, j);
        fprintf(stderr, "CPU: %f, GPU: %f\n", cpu_res[i * cols + j],
                gpu_res[i * cols + j]);
        return 1;
      }

  return 0;
}

// <--- GPU FUNCTIONS --->

/*
_add_matrices_gpu performs matrix addition on `m1` and `m2` and stores the
result in `res`.

It should not be used directly; instead, use the wrapper function
`add_matrices_gpu`.
*/
__global__ void _add_matrices_gpu(float_t *res, float_t *m1, float_t *m2,
                                  uint32_t rows, uint32_t cols) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= rows || y >= cols)
    return;

  uint32_t index = (x * cols) + y;

  res[index] = m1[index] + m2[index];
}

/*
add_matrices_gpu performs matrix addition on `m1` and `m2` and stores the result
in `res`.

It also performs CUDA memory management and kernel invocation, using the
specified `block_width` and `block_height` for the kernel, as well as timing the
operation.

It returns the time taken to nanosecond-precision.
*/
int64_t add_matrices_gpu(float_t *res, float_t *m1, float_t *m2, uint32_t rows,
                         uint32_t cols, uint32_t block_width,
                         uint32_t block_height) {
  float_t *d_res, *d_m1, *d_m2;

  cudaMalloc(&d_res, rows * cols * sizeof(float_t));
  cudaMalloc(&d_m1, rows * cols * sizeof(float_t));
  cudaMalloc(&d_m2, rows * cols * sizeof(float_t));

  cudaMemcpy(d_m1, m1, rows * cols * sizeof(float_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, m2, rows * cols * sizeof(float_t), cudaMemcpyHostToDevice);

  dim3 block(block_width, block_height);
  dim3 grid((rows + block_width - 1) / block_width,
            (cols + block_height - 1) / block_height);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _add_matrices_gpu<<<grid, block>>>(d_res, d_m1, d_m2, rows, cols);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaMemcpy(res, d_res, rows * cols * sizeof(float_t), cudaMemcpyDeviceToHost);

  cudaFree(d_res);
  cudaFree(d_m1);
  cudaFree(d_m2);

  return get_time_diff_ns(start, end);
}

// <--- TEST FUNCTIONS --->

/*
_print_matrix shows the contents of the matrix `m` in a human-readable format.
*/
void _print_matrix(float_t *m, uint32_t rows, uint32_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      printf("%f ", m[i * cols + j]);
    printf("\n");
  }
}

/*
verify_addition_result checks if the result of the addition operation between
`m1` and `m2` in `res` is correct.

If the result is correct, it returns 0. Otherwise, it returns 1.
*/
uint32_t _verify_addition_result(float_t *res, float_t *m1, float_t *m2,
                                 uint32_t rows, uint32_t cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (res[i * cols + j] != m1[i * cols + j] + m2[i * cols + j])
        return 1;

  return 0;
}

// <--- MAIN FUNCTION --->

int main(void) {
  // Seed the system RNG
  srand(time(NULL));

  // [[cols, rows]]
  uint32_t test_cases_matrices[5][2] = {
      {10, 10}, {100, 1000}, {1000, 1000}, {500, 2000}, {100, 10000}};

  // [[width, height]]
  uint32_t test_cases_blocks[3][2] = {{16, 16}, {16, 32}, {32, 16}};

  // Do a Warm-up run
  for (int i = 0; i < 2; i++) {
    // Test cases
    for (int j = 0; j < 5; j++) {
      if (i) {
        printf("Test Case #%d\n", j);
        printf("%d-by-%d matrix\n", test_cases_matrices[j][0],
               test_cases_matrices[j][1]);
      }
      // Matrix dimensions
      for (int k = 0; k < 3; k++) {
        uint32_t rows = test_cases_matrices[j][0];
        uint32_t cols = test_cases_matrices[j][1];

        float_t *m1 = (float_t *)malloc(rows * cols * sizeof(float_t));
        float_t *m2 = (float_t *)malloc(rows * cols * sizeof(float_t));
        float_t *cpu_res = (float_t *)malloc(rows * cols * sizeof(float_t));
        float_t *gpu_res = (float_t *)malloc(rows * cols * sizeof(float_t));

        populate_matrix(m1, rows, cols);
        populate_matrix(m2, rows, cols);

        int64_t cpu_time = add_matrices_cpu(cpu_res, m1, m2, rows, cols);
        int64_t gpu_time =
            add_matrices_gpu(gpu_res, m1, m2, rows, cols,
                             test_cases_blocks[k][0], test_cases_blocks[k][1]);

        if (i) {
          if (!k)
            printf("\tCPU time: %ld nanoseconds\n", cpu_time);
          printf("\tGPU (%d-by-%d block size): %ld nanoseconds\n",
                 test_cases_blocks[k][0], test_cases_blocks[k][1], gpu_time);
        }

        if (compare_outputs(cpu_res, gpu_res, rows, cols))
          return 1;

        free(m1);
        free(m2);
        free(cpu_res);
        free(gpu_res);
      }
    }
  }

  return 0;
}
