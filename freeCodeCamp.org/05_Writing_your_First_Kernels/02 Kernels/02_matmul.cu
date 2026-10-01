#include <cuda_runtime.h>

#include <cstdio>

#include "eva_cuda/check.cuh"

#define M 4096
#define K 2048
#define N 1024
#define BLOCK_SIZE 256

__global__ void matmal(const float* A, const float* B, float* C, const int m,
                       const int n, const int k) {}

int main() {
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  CUDA_CHECK(cudaMallocHost(&h_A, size_A));
  CUDA_CHECK(cudaMallocHost(&h_B, size_B));
  CUDA_CHECK(cudaMallocHost(&h_C, size_C));

  CUDA_CHECK(cudaMalloc(&d_A, size_A));
  CUDA_CHECK(cudaMalloc(&d_B, size_B));
  CUDA_CHECK(cudaMalloc(&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks(M + BLOCK_SIZE - 1 / BLOCK_SIZE, N + BLOCK_SIZE - 1 / BLOCK_SIZE);

  matmal<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));

  return 0;
}
