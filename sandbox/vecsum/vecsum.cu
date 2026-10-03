#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "eva_cuda/bench.cuh"
#include "eva_cuda/check.cuh"

#define BLOCK_SIZE 256

constexpr size_t N = 1 << 28;
constexpr int kWarmup = 3;
constexpr int kRepeat = 20;

__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// __global__ void vector_sum_gpu(const float* x, float* y, const int n) {
//   // 405.51 ms
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;

//   for (int idx = tid; idx < n; idx += gridDim.x * blockDim.x) {
//     atomicAdd(y, x[idx]);
//   }
// }

// __global__ void vector_sum_gpu(const float* x, float* y, const int n) {
//   // 12.38 ms
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;

//   for (int idx = tid; idx < n; idx += gridDim.x * blockDim.x) {
//     float val = warpReduceSum(x[idx]);
//     if (threadIdx.x % 32 == 0) atomicAdd(y, val);
//   }
// }

// __global__ void vector_sum_gpu(const float* x, float* y, const int n) {
//   // 1.34 ms
//   __shared__ float sdata[BLOCK_SIZE / 32];
//   int tid = threadIdx.x;
//   int gid = blockIdx.x * blockDim.x + tid;

//   for (int idx = gid; idx < n; idx += gridDim.x * blockDim.x) {
//     float val = warpReduceSum(x[idx]);
//     if (tid % 32 == 0) sdata[tid / 32] = val;

//     __syncthreads();

//     for (int i = BLOCK_SIZE / 64; i > 0; i /= 2) {
//       if (tid < i) {
//         sdata[tid] += sdata[tid + i];
//       }
//       __syncthreads();
//     }
//     if (tid == 0) atomicAdd(y, sdata[0]);
//   }
// }

// __global__ void vector_sum_gpu(const float* x, float* y, const int n) {
//   // 0.686 ms
//   __shared__ float sdata[BLOCK_SIZE / 32];
//   int tid = threadIdx.x;
//   int gid = blockIdx.x * blockDim.x + tid;
//   int line = tid % 32;
//   int wid = tid / 32;

//   float val = 0.f;
//   for (int idx = gid; idx < n; idx += gridDim.x * blockDim.x) {
//     val += x[idx];
//   }

//   val = warpReduceSum(val);
//   if (line == 0) sdata[wid] = val;

//   __syncthreads();

//   for (int i = BLOCK_SIZE / 64; i > 0; i /= 2) {
//     if (tid < i) {
//       sdata[tid] += sdata[tid + i];
//     }
//     __syncthreads();
//   }
//   if (tid == 0) atomicAdd(y, sdata[0]);
// }

template <int kBlockSize>
__global__ void vector_sum_gpu(const float* x, float* y, const int n) {
  // 0.686 ms
  static_assert(kBlockSize % 32 == 0, "Block size must be a multiple of 32");
  static_assert(kBlockSize <= 1024, "Block size must not exceed 1024");
  if (blockDim.x != kBlockSize) __trap();

  __shared__ float sdata[kBlockSize / 32];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int lane = tid % 32;
  int wid = tid / 32;

  float val = 0.f;
  for (size_t idx = gid; idx < n; idx += gridDim.x * blockDim.x) {
    val += x[idx];
  }

  val = warpReduceSum(val);
  if (lane == 0) sdata[wid] = val;

  __syncthreads();

  if (wid == 0) {
    val = lane < kBlockSize / 32 ? sdata[lane] : 0.f;
    val = warpReduceSum(val);
    if (lane == 0) atomicAdd(y, val);
  }
}

void vector_sum_cpu(const float* x, float* y, const int n) {
  // 109.48 ms
  double res = 0.0;
  for (int i = 0; i < n; i++) {
    res += x[i];
  }
  *y = res;
}

int main() {
  // init device and cuda runtime context
  CUDA_CHECK(cudaSetDevice(0));

  float *h_x, *h_y_gpu;
  float *d_x, *d_y;
  float y_cpu;
  float* h_y_cpu = &y_cpu;

  size_t size = N * sizeof(float);

  CUDA_CHECK(cudaMallocHost(&h_x, size));
  CUDA_CHECK(cudaMallocHost(&h_y_gpu, sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)));

  // init x
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.f, 1000.f);

  for (size_t i = 0; i < N; i++) h_x[i] = dist(rng);

  CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

  // Calculate grid size
  int numSMs, blockPerSM;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blockPerSM, vector_sum_gpu<BLOCK_SIZE>, BLOCK_SIZE, 0));
  int blocks = numSMs * blockPerSM * 2;
  // int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("numSMs: %d, blockPerSM: %d\n", numSMs, blockPerSM);

  // benchmark gpu kernel
  L2Flusher l2;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> gpu_ms;
  gpu_ms.reserve(kRepeat);

  for (int i = -kWarmup; i < kRepeat; i++) {
    // reset accumulator and evict x from l2
    CUDA_CHECK(cudaMemsetAsync(d_y, 0, sizeof(float)));
    l2.flush();

    CUDA_CHECK(cudaEventRecord(start));
    vector_sum_gpu<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_x, d_y, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    if (i >= 0) {
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      gpu_ms.push_back(ms);
    }
  }
  CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(gpu_ms.begin(), gpu_ms.end());
  float gpu_median_ms = gpu_ms[kRepeat / 2];
  printf("Median GPU time: %f ms\n", gpu_median_ms);

  std::vector<double> cpu_ms;
  cpu_ms.reserve(kRepeat);
  for (int i = -kWarmup; i < kRepeat; i++) {
    auto t0 = std::chrono::steady_clock::now();
    vector_sum_cpu(h_x, h_y_cpu, N);
    auto t1 = std::chrono::steady_clock::now();

    if (i >= 0) {
      cpu_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(cpu_ms.begin(), cpu_ms.end());
  double cpu_median_ms = cpu_ms[kRepeat / 2];
  printf("Median CPU time: %f ms\n", cpu_median_ms);
  printf("Speedup: %f\n", cpu_median_ms / gpu_median_ms);

  printf("GPU result: %f\n", *h_y_gpu);
  printf("CPU result: %f\n", y_cpu);

  if (std::fabs(*h_y_gpu - y_cpu) > 1e-4 * std::fmax(1.0, std::fabs(y_cpu))) {
    printf("Mismatch: GPU = %f, CPU = %f\n", *h_y_gpu, y_cpu);
  } else {
    printf("Match: GPU = %f, CPU = %f\n", *h_y_gpu, y_cpu);
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  CUDA_CHECK(cudaFreeHost(h_x));
  CUDA_CHECK(cudaFreeHost(h_y_gpu));

  return 0;
}
