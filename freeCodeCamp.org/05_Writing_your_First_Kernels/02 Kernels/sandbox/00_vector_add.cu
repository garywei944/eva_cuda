/*
 * Copyright (c) 2026 Gary Wei
 *
 * Licensed under the MIT
 * See LICENSE file in the project root for full license information.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000000
#define BLOCK_SIZE 256

__global__ void vector_add_gpu(const float* a, const float* b, float* c,
                               int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  c[tid] = a[tid] + b[tid];
}

void init_vector(float* vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

int main() {
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
  float *d_a, *d_b, *d_c;

  size_t size = N * sizeof(float);

  h_a = (float*)malloc(size);
  h_b = (float*)malloc(size);
  h_c_cpu = (float*)malloc(size);
  h_c_gpu = (float*)malloc(size);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  srand(time(NULL));

  init_vector(h_a, N);
  init_vector(h_b, N);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
