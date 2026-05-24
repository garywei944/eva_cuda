/*
 * Copyright (c) 2026 Gary Wei
 *
 * Licensed under the MIT
 * See LICENSE file in the project root for full license information.
 */

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(err), cudaGetErrorString(err), func);
    exit(EXIT_FAILURE);
  }
}

__global__ void kernel1(float *data, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    data[tid] *= 2.0f;
  }
}
__global__ void kernel2(float *data, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    data[tid] += 1.0f;
  }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError status,
                                void *userData) {
  printf("Stream callback: Operation completed\n");
}

int main(void) {
  const int N = 1000000;
  size_t size = N * sizeof(float);
  float *h_data, *d_data;
  cudaStream_t stream1, stream2;
  cudaEvent_t event;
  std::cout << event << std::endl;

  // allocate memory
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_data, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, size));

  // init array
  for (int i = 0; i < N; i++) {
    h_data[i] = static_cast<float>(i);
  }

  // create streams with different priority
  int leastPriority, greatestPriority;
  CHECK_CUDA_ERROR(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  std::cout << "least, greatest priority: " << leastPriority << " "
            << greatestPriority << std::endl;
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking,
                                                leastPriority));
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking,
                                                greatestPriority));

  // create event
  CHECK_CUDA_ERROR(cudaEventCreate(&event));
  std::cout << event << std::endl;

  int blockDim = 256;
  int gridDim = (N + blockDim - 1) / blockDim;

  CHECK_CUDA_ERROR(cudaEventDestroy(event));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

  // free memory
  CHECK_CUDA_ERROR(cudaFreeHost(h_data));
  CHECK_CUDA_ERROR(cudaFree(d_data));

  return 0;
}
