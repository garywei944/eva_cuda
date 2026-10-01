#include <cuda_runtime.h>

#include <cstdio>

#include "eva_cuda/check.cuh"

int main() {
  int h_counter_no_atomic = 0;
  int h_counter_atomic = 0;
  int *d_counter_no_atomic, *d_counter_atomic;

  CUDA_CHECK(cudaMalloc(&d_counter_no_atomic, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counter_atomic, sizeof(int)));

  CUDA_CHECK(cudaFree(d_counter_no_atomic));
  CUDA_CHECK(cudaFree(d_counter_atomic));

  return 0;
}
