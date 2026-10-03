#ifndef INCLUDE_EVA_CUDA_BENCH_CUH_
#define INCLUDE_EVA_CUDA_BENCH_CUH_

#include <cuda_runtime.h>

#include <cstddef>

#include "eva_cuda/check.cuh"

class L2Flusher {
 public:
  L2Flusher() {
    // get device
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int l2 = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    bytes_ = 2 * static_cast<size_t>(l2);
    CUDA_CHECK(cudaMalloc(&buf_, bytes_));
  }
  ~L2Flusher() { cudaFree(buf_); }

  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  void flush(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemsetAsync(buf_, 0, bytes_, stream));
  }

 private:
  void* buf_ = nullptr;
  size_t bytes_ = 0;
};

#endif  // INCLUDE_EVA_CUDA_BENCH_CUH_
