#ifndef INCLUDE_EVA_CUDA_CHECK_CUH_
#define INCLUDE_EVA_CUDA_CHECK_CUH_

#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(expr_to_check)                                      \
  do {                                                                 \
    cudaError_t result = expr_to_check;                                \
    if (result != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, \
              __LINE__, result, cudaGetErrorString(result));           \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#endif  // INCLUDE_EVA_CUDA_CHECK_CUH_
