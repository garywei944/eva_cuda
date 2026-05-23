/*
 * Copyright (c) 2026 Gary Wei
 *
 * Licensed under the MIT
 * See LICENSE file in the project root for full license information.
 */

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#define BLOCK_SIZE 16

__global__ void matmulKernel(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;

  if (row < N && col < N) {
    for (int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

void matmul(float *A, float *B, float *C, int N) {
  nvtxRangePush("Matrix Multiplication");

  float *d_A, *d_B, *d_C;
  int size = N * N * sizeof(float);

  nvtxRangePush("MemoryAllocation");
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  nvtxRangePop();

  nvtxRangePush("Kernel Execution");
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y);

  matmulKernel<<<gridDim, blockDim>>>(A, B, C, N);
  cudaDeviceSynchronize();
  nvtxRangePop();

  nvtxRangePush("Memory Copy D2H");
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  nvtxRangePush("Memory Deallocation");
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  nvtxRangePop();

  nvtxRangePop();
}

int main() {
  const int N = 1024;

  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N];

  matmul(A, B, C, N);

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
