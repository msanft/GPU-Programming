#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>

enum Approach {
  ATOMIC,
  CASCADED,
};

// 40MB = 4 Bytes (float_t) * 10M elements
#define ELEM_COUNT (10 * 1024 * 1024)

#define BLOCK_SIZE 256

// Checked for the Tesla T4 GPU. 4 Blocks per SM as per:
// https://www.reddit.com/r/CUDA/comments/1ekin72/which_cuda_block_configuration_is_better_for/
#define NUM_BLOCKS 160

// <--- HELPER FUNCTIONS --->

/*
populate_vector fills the vector `res` with random float values.
Values are in the range [0, 25.5] to avoid precision issues in comparison.
*/
void populate_vector(float_t *res, size_t size) {
  for (size_t i = 0; i < size; i++)
    res[i] = (float_t)(rand() & 0xFF) / 10.0f;
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
compare_outputs checks if the CPU and GPU results match within a small allowed
error margin to account for precision errors.
*/
uint32_t compare_outputs(float_t cpu_res, float_t gpu_res) {
  float_t relative_error = fabs((cpu_res - gpu_res) / cpu_res);
  float_t epsilon = 1e-1f;
  if (relative_error > epsilon) {
    fprintf(stderr, "CPU and GPU results do not match.\n");
    fprintf(stderr, "CPU: %f, GPU: %f\n", cpu_res, gpu_res);
    fprintf(stderr, "Relative error: %f%%\n", relative_error * 100);
    return 1;
  }
  return 0;
}

// <--- CPU FUNCTIONS --->

/*
_reduce_vector_cpu computes the sum of all elements in the vector `v`.
*/
float_t _reduce_vector_cpu(float_t *v, size_t size) {
  float_t res = 0;
  for (size_t i = 0; i < size; i++)
    res += v[i];
  return res;
}

/*
reduce_vector_cpu computes the sum of all elements in the vector `v`.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t reduce_vector_cpu(float_t *res, float_t *v, size_t size) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  *res = _reduce_vector_cpu(v, size);

  clock_gettime(CLOCK_MONOTONIC, &end);
  return get_time_diff_ns(start, end);
}

// <--- GPU FUNCTIONS --->

/*
_reduce_vector_gpu_atomic performs vector reduction using global memory atomic
add.
*/
__global__ void _reduce_vector_gpu_atomic(float_t *res, float_t *v,
                                          size_t size) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  float_t sum = 0.0f;
  for (uint32_t i = tid; i < size; i += stride)
    sum += v[i];

  atomicAdd(res, sum);
}

/*
_reduce_vector_gpu_cascaded performs vector reduction using the cascaded
approach from the homework description.
*/
__global__ void _reduce_vector_gpu_cascaded(float_t *res, float_t *v,
                                            size_t size) {
  __shared__ float_t shared_sum[BLOCK_SIZE];

  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  float_t thread_sum = 0.0f;
  for (uint32_t i = gid; i < size; i += stride)
    thread_sum += v[i];

  shared_sum[tid] = thread_sum;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      shared_sum[tid] += shared_sum[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(res, shared_sum[0]);
}

/*
reduce_vector_gpu performs vector reduction using the given approach.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t reduce_vector_gpu(float_t *res, float_t *v, size_t size,
                          Approach approach) {
  float_t *d_res, *d_v;

  checkCudaErrors(cudaMalloc(&d_v, size * sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_res, sizeof(float_t)));

  checkCudaErrors(
      cudaMemcpy(d_v, v, size * sizeof(float_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_res, 0, sizeof(float_t)));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  dim3 block(BLOCK_SIZE);
  dim3 grid;

  switch (approach) {
  case ATOMIC:
    grid.x = (size + block.x - 1) / block.x;
    _reduce_vector_gpu_atomic<<<grid, block>>>(d_res, d_v, size);
    break;
  case CASCADED:
    grid.x = NUM_BLOCKS;
    _reduce_vector_gpu_cascaded<<<grid, block>>>(d_res, d_v, size);
    break;
  }

  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(
      cudaMemcpy(res, d_res, sizeof(float_t), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_res));

  return get_time_diff_ns(start, end);
}

// <--- MAIN FUNCTION --->

int main(void) {
  srand(time(NULL));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  printf("SM Count: %d\n", prop.multiProcessorCount);

  float_t *v = (float_t *)malloc(ELEM_COUNT * sizeof(float_t));
  float_t cpu_res, gpu_res, gpu_res_cascaded;
  int64_t cpu_time, gpu_time_atomic, gpu_time_cascaded;

  populate_vector(v, ELEM_COUNT);

  cpu_time = reduce_vector_cpu(&cpu_res, v, ELEM_COUNT);
  printf("CPU time: %ld ns\n", cpu_time);

  gpu_time_atomic = reduce_vector_gpu(&gpu_res, v, ELEM_COUNT, ATOMIC);
  printf("\nGPU time (atomic): %ld ns\n", gpu_time_atomic);

  if (compare_outputs(cpu_res, gpu_res))
    goto error;

  gpu_time_cascaded =
      reduce_vector_gpu(&gpu_res_cascaded, v, ELEM_COUNT, CASCADED);
  printf("\nGPU time (cascaded): %ld ns\n", gpu_time_cascaded);

  if (compare_outputs(cpu_res, gpu_res_cascaded))
    goto error;

  free(v);
  return 0;

error:
  free(v);
  return 1;
}
