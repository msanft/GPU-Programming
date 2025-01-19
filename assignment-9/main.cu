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

#define MAX_THREADS 512
#define VEC_MIN_SIZE 100000
#define VEC_MAX_SIZE 1000000
#define VEC_SIZE_STEP 100000

// <--- HELPER FUNCTIONS --->

/*
populate_vector fills the vector `res` with random integer values.
*/
void populate_vector(int32_t *res, size_t size) {
  for (size_t i = 0; i < size; i++)
    res[i] = rand() % 100;
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
compare_outputs checks if the CPU and GPU results match.
*/
uint32_t compare_outputs(int32_t *cpu_res, int32_t *gpu_res, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (cpu_res[i] != gpu_res[i]) {
      fprintf(stderr, "CPU and GPU results do not match at index %ld.\n", i);
      fprintf(stderr, "CPU: %d, GPU: %d\n", cpu_res[i], gpu_res[i]);
      return 1;
    }
  }
  return 0;
}

/*
best_block_size returns the best block size (power of 2) for a given vector size
`n`.
*/
uint32_t best_block_size(uint32_t n) {
  if (n <= 1)
    return 1;
  uint32_t p = 1;
  while (p <= n / 2) {
    p *= 2;
  }
  return (n > p) ? p * 2 : p;
}

// <--- CPU FUNCTIONS --->

/*
_scan_vector_cpu performs a prefix-sum scan of all elements in the vector `v`,
storing the result in the vector `res`.
*/
void _scan_vector_cpu(int32_t *res, int32_t *v, size_t size) {
  res[0] = v[0];
  for (size_t i = 1; i < size; i++)
    res[i] = res[i - 1] + v[i];
}

/*
scan_vector_cpu performs a prefix-sum scan of all elements in the vector `v`,
storing the result in the vector `res`.

It also times the operation and returns the time taken to nanosecond-precision.
*/
int64_t scan_vector_cpu(int32_t *res, int32_t *v, size_t size) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  _scan_vector_cpu(res, v, size);

  clock_gettime(CLOCK_MONOTONIC, &end);
  return get_time_diff_ns(start, end);
}

// <--- GPU FUNCTIONS --->

/*
_scan_vector_gpu performs a prefix-sum scan of all elements in the vector `v`,
storing the result in the vector `res`.

It uses the work-efficient approach by Brent and Kung, as described in the
assignment.
*/
__global__ void _scan_vector_gpu(int32_t *res, int32_t *v, int32_t *aux,
                                 size_t size, size_t aux_size) {
  extern __shared__ int32_t sdata[];

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
    sdata[threadIdx.x] = v[i];

  for (uint32_t stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();

    if (i >= size)
      continue;

    uint32_t idx = (threadIdx.x + 1) * stride * 2 - 1;

    if (idx < blockDim.x)
      sdata[idx] += sdata[idx - stride];
  }

  for (uint32_t stride = blockDim.x / 4; stride > 0; stride /= 2) {
    __syncthreads();

    uint32_t idx = (threadIdx.x + 1) * stride * 2 - 1;

    if ((idx + stride + blockIdx.x * blockDim.x) >= size)
      continue;

    if ((idx + stride) < blockDim.x)
      sdata[idx + stride] += sdata[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t remainder = size % blockDim.x;
    uint32_t idx = (blockIdx.x < gridDim.x - 1)
                       ? blockDim.x - 1
                       : (remainder == 0 ? blockDim.x - 1 : remainder - 1);

    aux[blockIdx.x] = sdata[idx];
  }

  __syncthreads();

  if (i < size)
    res[i] = sdata[threadIdx.x];
}

/*
_scan_vector_gpu_fixup performs the final step of the prefix-sum scan,
accounting for output vectors which don't fit into a single block.
*/
__global__ void _scan_vector_gpu_fixup(int32_t *res, int32_t *v, int32_t *aux,
                                       size_t size, size_t aux_size) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i >= size) || (blockIdx.x == 0))
    return;

  __shared__ int32_t block_sum;

  if (threadIdx.x != 0)
    goto finish;

  block_sum = 0;
  for (uint32_t j = 0; j < blockIdx.x; j++)
    block_sum += aux[j];

finish:
  __syncthreads();

  res[i] += block_sum;
}

/*
scan_vector_gpu performs a prefix-sum scan of all elements in the vector `v`,
storing the result in the vector `res`.

It uses the work-efficient approach by Brent and Kung, as described in the
assignment.
*/
int64_t scan_vector_gpu(int32_t *res, int32_t *v, size_t size) {
  int32_t *d_res, *d_v, *d_aux;

  dim3 block((size >= MAX_THREADS) ? MAX_THREADS : best_block_size(size), 1);
  dim3 grid(ceil(size / (float_t)block.x), 1);
  uint32_t aux_size = best_block_size(grid.x);

  checkCudaErrors(cudaMalloc(&d_v, size * sizeof(int32_t)));
  checkCudaErrors(cudaMalloc(&d_res, size * sizeof(int32_t)));
  checkCudaErrors(cudaMalloc(&d_aux, aux_size * sizeof(int32_t)));

  checkCudaErrors(
      cudaMemcpy(d_v, v, size * sizeof(int32_t), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(d_res, 0, size * sizeof(int32_t)));
  checkCudaErrors(cudaMemset(d_aux, 0, aux_size * sizeof(int32_t)));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  _scan_vector_gpu<<<grid, block, block.x * sizeof(int32_t)>>>(
      d_res, d_v, d_aux, size, aux_size);
  if (grid.x > 1)
    _scan_vector_gpu_fixup<<<grid, block>>>(d_res, d_v, d_aux, size, aux_size);

  checkCudaErrors(cudaDeviceSynchronize());

  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(
      cudaMemcpy(res, d_res, size * sizeof(int32_t), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_res));
  checkCudaErrors(cudaFree(d_aux));

  return get_time_diff_ns(start, end);
}

// <--- MAIN FUNCTION --->

int main(void) {
  srand(time(NULL));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("GPU: %s\n", prop.name);
  printf("SM Count: %d\n", prop.multiProcessorCount);
  int32_t *v, *cpu_res, *gpu_res;
  int64_t cpu_time, gpu_time;

  for (size_t size = VEC_MIN_SIZE; size <= VEC_MAX_SIZE;
       size += VEC_SIZE_STEP) {
    v = (int32_t *)malloc(size * sizeof(int32_t));
    cpu_res = (int32_t *)malloc(size * sizeof(int32_t));
    gpu_res = (int32_t *)malloc(size * sizeof(int32_t));

    printf("Vector size: %ld\n", size);

    populate_vector(v, size);

    cpu_time = scan_vector_cpu(cpu_res, v, size);
    printf("\tCPU time: %ld ns\n", cpu_time);

    gpu_time = scan_vector_gpu(gpu_res, v, size);
    printf("\tGPU time: %ld ns\n", gpu_time);

    if (compare_outputs(cpu_res, gpu_res, size))
      goto error;

    free(v);
    free(cpu_res);
    free(gpu_res);
  }
  return 0;
error:
  free(v);
  free(cpu_res);
  free(gpu_res);
  return 1;
}
