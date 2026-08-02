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

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32

void init_matrix(float *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = (float)rand() / RAND_MAX;
  }
}

int main() {
  float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
  float *d_A, *d_B, *d_C;

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  h_A = (float *)malloc(size_A);
  h_B = (float *)malloc(size_B);
  h_C_cpu = (float *)malloc(size_C);
  h_C_gpu = (float *)malloc(size_C);

  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  srand(time(NULL));
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);

  return 0;
}
