#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define NUM_ITERATIONS 100000000

// <--- HELPER FUNCTIONS --->

/*
get_time_diff_ns calculates the difference between two timespec structs in
nanoseconds.
*/
int64_t get_time_diff_ns(struct timespec start, struct timespec end) {
  int64_t diff = (int64_t)(end.tv_sec - start.tv_sec) * (int64_t)1e9 +
                 (int64_t)(end.tv_nsec - start.tv_nsec);
  return diff;
}

// <--- GPU FUNCTIONS --->

/*
_parallel_calculations performs a series of dummy calculations on a set of
variables, that should be influenced by IL parallelization.

The results are stored in `res`, even though it doesn't have any meaning, as
it's solely used to demonstrate the power of ILP.

It does not use global memory.

It should not be used directly. Instead, use the wrapper function
`parallel_calculations`.
*/
__global__ void _parallel_calculations(float_t *res_a, float_t *res_d,
                                       float_t *res_e, float_t *res_f) {
  // dummy values
  float_t a = 1.0f;
  float_t b = 2.0f;
  float_t c = 3.0f;
  float_t d = 4.0f;
  float_t e = 5.0f;
  float_t f = 6.0f;

#pragma unroll 16
  for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
    a = a * b + c;
    d = d * b + c;
    e = e * b + c;
    f = f * b + c;
  }

  res_a[threadIdx.x] = a;
  res_d[threadIdx.x] = d;
  res_e[threadIdx.x] = e;
  res_f[threadIdx.x] = f;
}

/*
_non_parallel_calculations performs a series of dummy calculations on a set of
variables, that should *not* be influenced by IL parallelization.

The results are stored in `res`, even though it doesn't have any meaning, as
it's solely used to demonstrate the power of ILP.

It does not use global memory.

It should not be used directly. Instead, use the wrapper function
`non_parallel_calculations`.
*/
__global__ void _non_parallel_calculations(float_t *res) {
  // dummy values
  float_t a = 1.0f;
  float_t b = 2.0f;
  float_t c = 3.0f;

#pragma unroll 16
  for (uint32_t i = 0; i < NUM_ITERATIONS; i++)
    a = a * b + c;

  res[threadIdx.x] = a;
}

/*
_memory_parallel_calculations performs a series of independent memory operations
that should be influenced by IL parallelization.

It uses global memory extensively to demonstrate memory ILP.

It should not be used directly. Instead, use the wrapper function
`memory_parallel_calculations`.
*/
__global__ void _memory_parallel_calculations(float_t *res_a, float_t *res_b,
                                              float_t *res_c, float_t *res_d) {
#pragma unroll 16
  for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
    res_a[blockIdx.x*blockDim.x+threadIdx.x] *= 3.0f;
    res_b[blockIdx.x*blockDim.x+threadIdx.x] *= 3.0f;
    res_c[blockIdx.x*blockDim.x+threadIdx.x] *= 3.0f;
    res_d[blockIdx.x*blockDim.x+threadIdx.x] *= 3.0f;
  }
}

/*
parallel_calculations is a wrapper function for `_parallel_calculations`, taking
care of CUDA memory management, timing, and CUDA kernel invocation.

It performs a series of dummy calculations on a set of variables, that should be
influenced by IL parallelization.

The results are stored in `res`, even though it doesn't have any meaning, as
it's solely used to demonstrate the power of ILP.

It does not use global memory.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t parallel_calculations(uint32_t block_width, uint32_t block_height) {
  float_t *d_res_a, *d_res_d, *d_res_e, *d_res_f;

  dim3 grid(1, 1);
  dim3 block(block_width, block_height);

  uint32_t thread_count = block_width * block_height;

  // a, d, e, f -> 4 results
  cudaMalloc(&d_res_a, sizeof(float_t) * thread_count);
  cudaMalloc(&d_res_d, sizeof(float_t) * thread_count);
  cudaMalloc(&d_res_e, sizeof(float_t) * thread_count);
  cudaMalloc(&d_res_f, sizeof(float_t) * thread_count);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _parallel_calculations<<<grid, block>>>(d_res_a, d_res_d, d_res_e, d_res_f);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaFree(d_res_a);
  cudaFree(d_res_d);
  cudaFree(d_res_e);
  cudaFree(d_res_f);

  return get_time_diff_ns(start, end);
}

/*
non_parallel_calculations is a wrapper function for
`_non_parallel_calculations`, taking care of CUDA memory management, timing, and
CUDA kernel invocation.

It performs a series of dummy calculations on a set of variables, that should
*not* be influenced by IL parallelization.

The results are stored in `res`, even though it doesn't have any meaning, as
it's solely used to demonstrate the power of ILP.

It does not use global memory.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t non_parallel_calculations(uint32_t block_width, uint32_t block_height) {
  float_t *d_res;

  dim3 grid(1, 1);
  dim3 block(block_width, block_height);

  uint32_t thread_count = block_width * block_height;

  // 1 result only
  cudaMalloc(&d_res, sizeof(float_t) * thread_count);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _non_parallel_calculations<<<grid, block>>>(d_res);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaFree(d_res);

  return get_time_diff_ns(start, end);
}

/*
memory_parallel_calculations is a wrapper function for
`_memory_parallel_calculations`.

It performs a series of independent memory operations that should be influenced
by IL parallelization and memory ILP.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t memory_parallel_calculations(uint32_t block_width,
                                     uint32_t block_height) {
  float_t *d_res_a, *d_res_b, *d_res_c, *d_res_d;

  dim3 grid(1, 1);
  dim3 block(block_width, block_height);

  uint32_t thread_count = block_width * block_height;

  cudaMalloc(&d_res_a, sizeof(float_t) * thread_count);
  cudaMalloc(&d_res_b, sizeof(float_t) * thread_count);
  cudaMalloc(&d_res_c, sizeof(float_t) * thread_count);
  cudaMalloc(&d_res_d, sizeof(float_t) * thread_count);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _memory_parallel_calculations<<<grid, block>>>(d_res_a, d_res_b, d_res_c,
                                                 d_res_d);
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaFree(d_res_a);
  cudaFree(d_res_b);
  cudaFree(d_res_c);
  cudaFree(d_res_d);

  return get_time_diff_ns(start, end);
}

// <--- MAIN FUNCTION --->

int main(void) {
  printf("Iterations: %d\n", NUM_ITERATIONS);

  uint32_t block_widths[] = {1, 2, 4, 8, 16, 32};
  uint32_t block_heights[] = {1, 2, 4, 8, 16, 32};

  for (uint32_t i = 0; i <= 5; i++) {
    printf("Block width: %d, block height: %d\n", block_widths[i],
           block_heights[i]);
    printf("Thread count: %d\n", block_widths[i] * block_heights[i]);

    int64_t ilp4_time =
        parallel_calculations(block_widths[i], block_heights[i]);
    // 4 operations per iteration * NUM_ITERATIONS
    float_t time_per_op_ilp = (float_t)ilp4_time / (4 * NUM_ITERATIONS);
    printf("\tILP4 time: %ld ns (%.2f ns per operation)\n", ilp4_time,
           time_per_op_ilp);

    int64_t non_ilp_time =
        non_parallel_calculations(block_widths[i], block_heights[i]);
    // 1 operation per iteration * NUM_ITERATIONS
    float_t time_per_op_non_ilp = (float_t)non_ilp_time / NUM_ITERATIONS;
    printf("\tNon-ILP time: %ld ns (%.2f ns per operation)\n", non_ilp_time,
           time_per_op_non_ilp);

    int64_t memory_ilp_time =
        memory_parallel_calculations(block_widths[i], block_heights[i]);
    // 4 operations per iteration * NUM_ITERATIONS
    float_t time_per_op_memory_ilp =
        (float_t)memory_ilp_time / (4 * NUM_ITERATIONS);
    printf("\tMemory ILP time: %ld ns (%.2f ns per operation)\n",
           memory_ilp_time, time_per_op_memory_ilp);

    float_t speedup = (float_t)time_per_op_non_ilp / time_per_op_ilp;
    printf("\tSpeedup ILP4 vs. non-ILP: %.2f%%\n", (speedup - 1) * 100);

    float_t memory_speedup =
        (float_t)time_per_op_non_ilp / time_per_op_memory_ilp;
    printf("\tSpeedup Memory ILP vs. non-ILP: %.2f%%\n",
           (memory_speedup - 1) * 100);
  }

  checkCudaErrors(cudaDeviceReset());
  return 0;
}
