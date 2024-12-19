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

// 40MB = 4 Bytes (float_t) * 10M size
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

/*
max_pow_2 returns the next power of 2 greater than or equal to `n`.
*/
uint32_t max_pow_2(uint32_t n) {
  uint32_t res = 1;
  while (res * 2 <= n) {
      res *= 2;
  }
  return res;
}

// <--- CPU FUNCTIONS --->

/*
_reduce_vector_cpu computes the sum of all size in the vector `v`.
*/
float_t _reduce_vector_cpu(float_t *v, size_t size) {
  float_t res = 0;
  for (size_t i = 0; i < size; i++)
    res += v[i];
  return res;
}

/*
reduce_vector_cpu computes the sum of all size in the vector `v`.

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
warp_reduce performs a warp-level reduction on the shared memory array `shared_sum`.
*/
template<uint32_t block_size>
__device__ void warp_reduce(volatile float_t *shared_sum, uint32_t tid)
{
    if(block_size >= 64)
      shared_sum[tid] += shared_sum[tid + 32];
    if(block_size >= 32)
      shared_sum[tid] += shared_sum[tid + 16];
    if(block_size >= 16)
      shared_sum[tid] += shared_sum[tid + 8];
    if(block_size >= 8)
      shared_sum[tid] += shared_sum[tid + 4];
    if(block_size >= 4)
      shared_sum[tid] += shared_sum[tid + 2];
    if(block_size >= 2)
      shared_sum[tid] += shared_sum[tid + 1];
}

/*
_reduce_vector_gpu_cascaded performs vector reduction using the cascaded
approach from Harris.
*/
template<uint32_t block_size>
__global__ void _reduce_vector_gpu_cascaded(float_t *res, float_t *v,
                                            size_t size) {
  extern __shared__ float shared_sum[];

  uint32_t tid = threadIdx.x;
  uint32_t gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
  uint32_t stride = BLOCK_SIZE * 2 * gridDim.x;
  float_t thread_sum = 0.0f;

  while (gid < size) {
    thread_sum += v[gid];
    if (gid + BLOCK_SIZE < size)
      thread_sum += v[gid + BLOCK_SIZE];
    gid += stride;
  }
  shared_sum[tid] = thread_sum;
  __syncthreads();

  for (uint32_t s = blockDim.x / 32; s > 32; s >>= 1) {
    if (tid < s)
      shared_sum[tid] += shared_sum[tid + s];
    __syncthreads();
  }

  if(block_size >= 512) {
		if(tid < 256)
			shared_sum[tid] += shared_sum[tid + 256];
		__syncthreads();
	}

	if(block_size >= 256) {
		if(tid < 128)
			shared_sum[tid] += shared_sum[tid + 128];
		__syncthreads();
	}

	if(block_size >= 128) {
		if(tid < 64)
			shared_sum[tid] += shared_sum[tid + 64];
		__syncthreads();
	}

  if (tid < 32)
    warp_reduce<block_size>(shared_sum, tid);

  if (tid == 0)
    res[blockIdx.x] = shared_sum[0];
}

/*
reduce_vector_gpu performs vector reduction using the given approach.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t reduce_vector_gpu(float_t *res, float_t *v, size_t size) {
  float_t *d_res, *d_v;
  bool is_final_output_in_res = true;

  checkCudaErrors(cudaMalloc(&d_v, size * sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_res, sizeof(float_t)));

  checkCudaErrors(
      cudaMemcpy(d_v, v, size * sizeof(float_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_res, 0, sizeof(float_t)));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  dim3 block(BLOCK_SIZE);
  dim3 grid;

  grid.x = NUM_BLOCKS;

  while (size > 1) {
    uint32_t shared_mem = block.x * sizeof(float_t);

    switch(block.x) {
      case 512:
        _reduce_vector_gpu_cascaded<512><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 256:
        _reduce_vector_gpu_cascaded<256><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 128:
        _reduce_vector_gpu_cascaded<128><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 64:
        _reduce_vector_gpu_cascaded<64><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 32:
        _reduce_vector_gpu_cascaded<32><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 16:
        _reduce_vector_gpu_cascaded<16><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 8:
        _reduce_vector_gpu_cascaded<8><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 4:
        _reduce_vector_gpu_cascaded<4><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 2:
        _reduce_vector_gpu_cascaded<2><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
      case 1:
        _reduce_vector_gpu_cascaded<1><<<grid, block, shared_mem>>>(d_res, d_v, size);
        break;
    }

    checkCudaErrors(cudaDeviceSynchronize());

		size = grid.x;
    block.x = min(max_pow_2(size), BLOCK_SIZE);
    grid.x = ceil(size / (float_t) block.x);

    // swap d_v and d_res
    float_t* temp = d_v;
    d_v = d_res;
    d_res = temp;
    is_final_output_in_res = !is_final_output_in_res;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(
      cudaMemcpy(res, is_final_output_in_res ? d_res : d_v, sizeof(float_t), cudaMemcpyDeviceToHost));

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
  float_t cpu_res, gpu_res_cascaded;
  int64_t cpu_time, gpu_time_cascaded;

  populate_vector(v, ELEM_COUNT);

  cpu_time = reduce_vector_cpu(&cpu_res, v, ELEM_COUNT);
  printf("CPU time: %ld ns\n", cpu_time);

  gpu_time_cascaded =
      reduce_vector_gpu(&gpu_res_cascaded, v, ELEM_COUNT);
  printf("\nGPU time (cascaded): %ld ns\n", gpu_time_cascaded);

  if (compare_outputs(cpu_res, gpu_res_cascaded))
    goto error;

  free(v);
  return 0;

error:
  free(v);
  return 1;
}
