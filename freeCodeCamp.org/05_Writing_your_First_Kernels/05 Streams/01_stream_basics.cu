/*
 * Copyright (c) 2026 Gary Wei
 *
 * Licensed under the MIT
 * See LICENSE file in the project root for full license information.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    C[tid] = A[tid] + B[tid];
  }
}

int main(void) {
  int N = 50000;
  size_t size = N * sizeof(float);

  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  cudaStream_t stream1, stream2;

  // init host arrays
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  for (int i = 0; i < N; i++) {
    h_A[i] = rand_r() / (float)RAND_MAX;
    h_B[i] = rand_r() / (float)RAND_MAX;
  }

  // init device arrays
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

  CHECK_CUDA_ERROR(
      cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));
  CHECK_CUDA_ERROR(
      cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));

  // sync stream2 before launch kernel
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

  int blockDim = 256;
  int gridDim = (N + blockDim - 1) / blockDim;
  vectorAdd<<<gridDim, blockDim, 0, stream1>>>(d_A, d_B, d_C, N);

  // copy result back
  CHECK_CUDA_ERROR(
      cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

  for (int i = 0; i < N; i++) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test passed!\n");

  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
