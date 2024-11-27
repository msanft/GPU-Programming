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

// Thread block dimensions to test
#define BLOCK_8 8
#define BLOCK_16 16
#define BLOCK_32 32

// Maximum filter size supported
#define MAX_FILTER_SIZE 25

enum Approach {
  GLOBAL_MEM,
  CONSTANT_MEM,
  TEXTURE_MEM,
};

// Constant memory for filter coefficients
__constant__ float_t d_filter_const[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

// <--- HELPER FUNCTIONS --->

/*
populate_matrix fills the matrix `res` with random float values.
Values are in the range [0, 25.5] to avoid precision issues in comparison.
*/
void populate_matrix(float_t *res, uint32_t width, uint32_t height) {
  for (uint32_t i = 0; i < height; i++)
    for (uint32_t j = 0; j < width; j++)
      res[i * width + j] = (float_t)(rand() & 0xFF) / 10.0f;
}

/*
populate_filter fills the filter with sample values.
For simplicity, we use a basic box filter normalized to sum to 1.
*/
void populate_filter(float_t *filter, uint32_t size) {
  float_t value = 1.0f / (size * size);
  for (uint32_t i = 0; i < size * size; i++)
    filter[i] = value;
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
effective_bandwidth calculates the effective bandwidth of a kernel in GB/s.
*/
float_t effective_bandwidth(int64_t nsec, uint32_t width, uint32_t height,
                            uint32_t filter_size) {
  float_t total_bytes = (float_t)width * height * sizeof(float) * 2.0f +
                        (float_t)filter_size * filter_size * sizeof(float);

  float_t seconds = (float_t)nsec / 1e9f;
  return (total_bytes / 1e9f) / seconds;
}

// <--- CPU FUNCTIONS --->

/*
_convolution_2d_cpu performs 2D convolution on CPU.
Ghost cells are treated as 0.
*/
void _convolution_2d_cpu(float_t *res, float_t *m, float_t *filter,
                         uint32_t height, uint32_t width,
                         uint32_t filter_size) {
  int32_t filter_radius = filter_size / 2;

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      float_t sum = 0.0f;

      for (int32_t fy = -filter_radius; fy <= filter_radius; fy++) {
        for (int32_t fx = -filter_radius; fx <= filter_radius; fx++) {
          int32_t in_y = y + fy;
          int32_t in_x = x + fx;

          if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
            float_t m_value = m[in_y * width + in_x];
            float_t filter_value = filter[(fy + filter_radius) * filter_size +
                                          (fx + filter_radius)];
            sum += m_value * filter_value;
          }
        }
      }

      res[y * width + x] = sum;
    }
  }
}

/*
convolution_2d_cpu is a wrapper that performs convolution and timing.
*/
int64_t convolution_2d_cpu(float_t *res, float_t *m, float_t *filter,
                           uint32_t height, uint32_t width,
                           uint32_t filter_size) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  _convolution_2d_cpu(res, m, filter, height, width, filter_size);

  clock_gettime(CLOCK_MONOTONIC, &end);
  return get_time_diff_ns(start, end);
}

// <--- GPU KERNELS --->

/*
_convolution_2d_gpu_global performs 2D convolution using global memory for the
filter.
*/
__global__ void _convolution_2d_gpu_global(float_t *res, float_t *m,
                                           float_t *filter, uint32_t height,
                                           uint32_t width,
                                           uint32_t filter_size) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int32_t filter_radius = filter_size / 2;
  float_t sum = 0.0f;

  for (int32_t fy = -filter_radius; fy <= filter_radius; fy++) {
    for (int32_t fx = -filter_radius; fx <= filter_radius; fx++) {
      int32_t in_y = y + fy;
      int32_t in_x = x + fx;

      if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
        float_t m_value = m[in_y * width + in_x];
        float_t filter_value =
            filter[(fy + filter_radius) * filter_size + (fx + filter_radius)];
        sum += m_value * filter_value;
      }
    }
  }

  res[y * width + x] = sum;
}

/*
_convolution_2d_gpu_constant performs 2D convolution using constant memory for
the filter.
*/
__global__ void _convolution_2d_gpu_constant(float_t *res, float_t *m,
                                             uint32_t height, uint32_t width,
                                             uint32_t filter_size) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int32_t filter_radius = filter_size / 2;
  float_t sum = 0.0f;

  for (int32_t fy = -filter_radius; fy <= filter_radius; fy++) {
    for (int32_t fx = -filter_radius; fx <= filter_radius; fx++) {
      int32_t in_y = y + fy;
      int32_t in_x = x + fx;

      if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
        float_t m_value = m[in_y * width + in_x];
        float_t filter_value =
            d_filter_const[(fy + filter_radius) * filter_size +
                           (fx + filter_radius)];
        sum += m_value * filter_value;
      }
    }
  }

  res[y * width + x] = sum;
}

/*
_convolution_2d_gpu_texture performs 2D convolution using texture memory for
m.
*/
__global__ void _convolution_2d_gpu_texture(float_t *res,
                                            cudaTextureObject_t tex,
                                            uint32_t height, uint32_t width,
                                            uint32_t filter_size) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int32_t filter_radius = filter_size / 2;
  float_t sum = 0.0f;

  for (int32_t fy = -filter_radius; fy <= filter_radius; fy++) {
    for (int32_t fx = -filter_radius; fx <= filter_radius; fx++) {
      int32_t in_y = y + fy;
      int32_t in_x = x + fx;

      if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
        float_t m_value = tex2D<float>(tex, in_x, in_y);
        float_t filter_value =
            d_filter_const[(fy + filter_radius) * filter_size +
                           (fx + filter_radius)];
        sum += m_value * filter_value;
      }
    }
  }

  res[y * width + x] = sum;
}

/*
convolution_2d_gpu is a wrapper that handles memory management and kernel
execution.
*/
int64_t convolution_2d_gpu(float_t *res, float_t *m, float_t *filter,
                           uint32_t height, uint32_t width,
                           uint32_t filter_size, uint32_t block_size,
                           Approach approach) {
  float_t *d_res, *d_m, *d_filter;
  cudaArray_t cuArray = nullptr;
  struct timespec start, end;

  checkCudaErrors(cudaMalloc(&d_res, height * width * sizeof(float_t)));
  checkCudaErrors(cudaMalloc(&d_m, height * width * sizeof(float_t)));

  checkCudaErrors(cudaMemcpy(d_m, m, height * width * sizeof(float_t),
                             cudaMemcpyHostToDevice));

  dim3 block(block_size, block_size);
  dim3 grid(ceil(width / (float)block_size), ceil(height / (float)block_size));

  cudaTextureObject_t tex = 0;
  if (approach == GLOBAL_MEM) {
    checkCudaErrors(
        cudaMalloc(&d_filter, filter_size * filter_size * sizeof(float_t)));
    checkCudaErrors(cudaMemcpy(d_filter, filter,
                               filter_size * filter_size * sizeof(float_t),
                               cudaMemcpyHostToDevice));
  } else if (approach == CONSTANT_MEM) {
    checkCudaErrors(cudaMemcpyToSymbol(
        d_filter_const, filter, filter_size * filter_size * sizeof(float_t)));
  } else if (approach == TEXTURE_MEM) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));

    checkCudaErrors(cudaMemcpy2DToArray(
        cuArray, 0, 0, m, width * sizeof(float_t), width * sizeof(float_t),
        height, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    checkCudaErrors(cudaMemcpyToSymbol(
        d_filter_const, filter, filter_size * filter_size * sizeof(float_t)));
  }

  clock_gettime(CLOCK_MONOTONIC, &start);

  switch (approach) {
  case GLOBAL_MEM:
    _convolution_2d_gpu_global<<<grid, block>>>(d_res, d_m, d_filter, height,
                                                width, filter_size);
    break;
  case CONSTANT_MEM:
    _convolution_2d_gpu_constant<<<grid, block>>>(d_res, d_m, height, width,
                                                  filter_size);
    break;
  case TEXTURE_MEM:
    _convolution_2d_gpu_texture<<<grid, block>>>(d_res, tex, height, width,
                                                 filter_size);
    break;
  }

  checkCudaErrors(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);

  checkCudaErrors(cudaMemcpy(res, d_res, height * width * sizeof(float_t),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_res));
  checkCudaErrors(cudaFree(d_m));
  if (approach == GLOBAL_MEM) {
    checkCudaErrors(cudaFree(d_filter));
  } else if (approach == TEXTURE_MEM) {
    if (tex != 0) {
      checkCudaErrors(cudaDestroyTextureObject(tex));
    }
    if (cuArray != nullptr) {
      checkCudaErrors(cudaFreeArray(cuArray));
    }
  }

  return get_time_diff_ns(start, end);
}

/*
compare_ress checks if the matrices match within a small epsilon.
*/
uint32_t compare_ress(float_t *cpu_res, float_t *gpu_res, uint32_t width,
                      uint32_t height) {
  const float_t epsilon = 1e-5;
  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j++) {
      float_t diff = fabs(cpu_res[i * width + j] - gpu_res[i * width + j]);
      if (diff > epsilon) {
        fprintf(stderr, "Results do not match at (%d, %d). ", i, j);
        fprintf(stderr, "CPU: %f, GPU: %f\n", cpu_res[i * width + j],
                gpu_res[i * width + j]);
        return 1;
      }
    }
  }
  return 0;
}

// <--- MAIN FUNCTION --->

int main(void) {
  srand(time(NULL));

  const uint32_t width = 4096;
  const uint32_t height = 2048;

  const uint32_t filter_sizes[] = {7, 13, 25};
  const uint32_t num_filter_sizes = sizeof(filter_sizes) / sizeof(uint32_t);

  const uint32_t block_sizes[] = {8, 16, 32};
  const uint32_t num_block_sizes = sizeof(block_sizes) / sizeof(uint32_t);

  float_t *m = (float_t *)malloc(height * width * sizeof(float_t));
  float_t *cpu_res = (float_t *)malloc(height * width * sizeof(float_t));
  float_t *gpu_res = (float_t *)malloc(height * width * sizeof(float_t));

  populate_matrix(m, width, height);

  for (uint32_t f = 0; f < num_filter_sizes; f++) {
    uint32_t filter_size = filter_sizes[f];
    printf("\nTesting with %dx%d filter:\n", filter_size, filter_size);
    printf("----------------------------------------\n");

    float_t *filter =
        (float_t *)malloc(filter_size * filter_size * sizeof(float_t));
    populate_filter(filter, filter_size);

    int64_t cpu_time =
        convolution_2d_cpu(cpu_res, m, filter, height, width, filter_size);
    printf("CPU time: %ld ns\n", cpu_time);
    printf("CPU effective bandwidth: %.2f GB/s\n",
           effective_bandwidth(cpu_time, width, height, filter_size));

    for (uint32_t b = 0; b < num_block_sizes; b++) {
      uint32_t block_size = block_sizes[b];
      printf("\nBlock size %dx%d:\n", block_size, block_size);
      printf("----------------\n");

      int64_t gpu_time_global =
          convolution_2d_gpu(gpu_res, m, filter, height, width, filter_size,
                             block_size, GLOBAL_MEM);
      printf("Global memory time: %ld ns\n", gpu_time_global);
      printf("Global memory effective bandwidth: %.2f GB/s\n",
             effective_bandwidth(gpu_time_global, width, height, filter_size));
      if (compare_ress(cpu_res, gpu_res, width, height))
        goto error;

      int64_t gpu_time_constant =
          convolution_2d_gpu(gpu_res, m, filter, height, width, filter_size,
                             block_size, CONSTANT_MEM);
      printf("Constant memory time: %ld ns\n", gpu_time_constant);
      printf(
          "Constant memory effective bandwidth: %.2f GB/s\n",
          effective_bandwidth(gpu_time_constant, width, height, filter_size));
      if (compare_ress(cpu_res, gpu_res, width, height))
        goto error;

      int64_t gpu_time_texture =
          convolution_2d_gpu(gpu_res, m, filter, height, width, filter_size,
                             block_size, TEXTURE_MEM);
      printf("Texture memory time: %ld ns\n", gpu_time_texture);
      printf("Texture memory effective bandwidth: %.2f GB/s\n",
             effective_bandwidth(gpu_time_texture, width, height, filter_size));
      if (compare_ress(cpu_res, gpu_res, width, height))
        goto error;
    }

    free(filter);
  }

  free(m);
  free(cpu_res);
  free(gpu_res);
  return 0;

error:
  free(m);
  free(cpu_res);
  free(gpu_res);
  return 1;
}
