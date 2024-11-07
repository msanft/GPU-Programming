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

It should not be used directly. Instead, use the wrapper function
`parallel_calculations`.
*/
__global__ void _parallel_calculations(float_t *res) {
  // dummy values
  float_t a = 1.0;
  float_t b = 2.0;
  float_t c = 3.0;
  float_t d = 4.0;
  float_t e = 5.0;
  float_t f = 6.0;

#pragma unroll 16
  for (unsigned int i = 0; i < NUM_ITERATIONS; i++) {
    a = a * b + c;
    d = d * b + c;
    e = e * b + c;
    f = f * b + c;
  }

  *(res) = a;
  *(res + 1) = d;
  *(res + 2) = e;
  *(res + 3) = f;
}

/*
_non_parallel_calculations performs a series of dummy calculations on a set of
variables, that should *not* be influenced by IL parallelization.

The results are stored in `res`, even though it doesn't have any meaning, as
it's solely used to demonstrate the power of ILP.

It should not be used directly. Instead, use the wrapper function
`non_parallel_calculations`.
*/
__global__ void _non_parallel_calculations(float_t *res) {
  // dummy values
  float_t a = 1.0;
  float_t b = 2.0;
  float_t c = 3.0;

#pragma unroll 16
  for (unsigned int i = 0; i < NUM_ITERATIONS; i++)
    a = a * b + c;

  *(res) = a;
}

/*
parallel_calculations is a wrapper function for `_parallel_calculations`, taking
care of CUDA memory management, timing, and CUDA kernel invocation.

It performs a series of dummy calculations on a set of variables, that should be
influenced by IL parallelization.

The results are stored in `res`, even though it doesn't have any meaning, as
it's solely used to demonstrate the power of ILP.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t parallel_calculations(uint32_t thread_count) {
  float_t *d_res;

  // a, d, e, f -> 4 results
  cudaMalloc(&d_res, 4 * sizeof(float_t));

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _parallel_calculations<<<1, thread_count>>>(d_res);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaFree(d_res);

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

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t non_parallel_calculations(uint32_t thread_count) {
  float_t *d_res;

  // 1 result only
  cudaMalloc(&d_res, sizeof(float_t));

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  _non_parallel_calculations<<<1, thread_count>>>(d_res);

  // Wait for all threads to finish.
  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  cudaFree(d_res);

  return get_time_diff_ns(start, end);
}

// <--- MAIN FUNCTION --->

int main(void) {
  for (uint32_t i = 0; i <= 11; i++) {
    uint32_t thread_count = 1 << i;
    printf("Thread count: %d\n", thread_count);

    int64_t ops_count_non_ilp = NUM_ITERATIONS * thread_count * 2;
    int64_t ops_count_ilp = ops_count_non_ilp * 4;

    int64_t ilp4_time = parallel_calculations(thread_count);

    printf("\tILP4 time: %ld ns\n", ilp4_time / ops_count_ilp);

    int64_t non_ilp_time = non_parallel_calculations(thread_count);

    printf("\tNon-ILP time: %ld ns\n", non_ilp_time / ops_count_non_ilp);

    int64_t speedup = non_ilp_time / ilp4_time;
    printf("\tSpeedup ILP4 vs. non-ILP: %ld%\n", (speedup - 1) * 100);
  }

  cudaDeviceReset();

  return 0;
}
