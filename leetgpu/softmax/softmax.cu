#include <cooperative_groups.h>
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

namespace cg = cooperative_groups;

// constexpr size_t N = 1 << 26;
constexpr size_t N = 1 << 10;
// Debugging: run GPU and CPU once each. Benchmark: kWarmup 3, kRepeat 20.
constexpr int kWarmup = 0;
constexpr int kRepeat = 1;

__inline__ __device__ float warpReduceMax(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16, 32));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8, 32));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4, 32));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2, 32));
  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1, 32));
  return val;
}

__inline__ __device__ float warpReduceSum(float val) {
  val += __shfl_xor_sync(0xffffffff, val, 16, 32);
  val += __shfl_xor_sync(0xffffffff, val, 8, 32);
  val += __shfl_xor_sync(0xffffffff, val, 4, 32);
  val += __shfl_xor_sync(0xffffffff, val, 2, 32);
  val += __shfl_xor_sync(0xffffffff, val, 1, 32);
  return val;
}

__inline__ __device__ float2 merge(float2 a, float2 b) {
  if (a.x == -INFINITY && b.x == -INFINITY) return a;

  // assume a.x larger
  if (a.x < b.x) {
    float2 t = a;
    a = b;
    b = t;
  }
  if (b.x == -INFINITY) return a;
  a.y += b.y * expf(b.x - a.x);
  return a;
}

__inline__ __device__ float2 warpMergeMaxSum(float2* input, int size) {
  int lane = threadIdx.x % 32;

  float local_max = lane < size ? input[lane].x : -INFINITY;
  float local_sum;
  float global_max = warpReduceMax(local_max);
  if (local_max == -INFINITY)
    local_sum = 0.f;
  else
    local_sum = input[lane].y / expf(global_max - local_max);
  float global_sum = warpReduceSum(local_sum);
  return make_float2(global_max, global_sum);
}

__inline__ __device__ float2 blockMergeMaxSum(float2* input, int input_size,
                                              float2* sdata, int sdata_size) {
  int tid = threadIdx.x;
  int lane = tid % 32;
  int wid = tid / 32;

  // block stride loop
  float2 local = make_float2(-INFINITY, 0.f);
  for (int i = tid; i < input_size; i += blockDim.x) {
    local = merge(local, input[i]);
  }
  float warp_max = warpReduceMax(local.x);
  if (local.y != 0.f) local.y /= expf(warp_max - local.x);
  float warp_sum = warpReduceSum(local.y);
  if (lane == 0) sdata[wid] = make_float2(warp_max, warp_sum);

  __syncthreads();

  return warpMergeMaxSum(sdata, sdata_size);
}

// TODO(garywei944): kernels
template <int threadsPerBlock>
__global__ void softmax_kernel(const float* x, float* y, float2* global_buf,
                               const int n) {
  // all warp sum and warp max.
  __shared__ float2 sdata[threadsPerBlock / 32];

  auto grid = cg::this_grid();

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;
  int lane = tid % 32;
  int wid = tid / 32;
  int stride = gridDim.x * blockDim.x;

  // 1. grid stride loop and reduce to get #blocks max and scaled sum
  float2 local = make_float2(-INFINITY, 0.f);
  for (int i = gid; i < n; i += stride) {
    local = merge(local, make_float2(x[i], 1.f));
  }
  // warp local max
  float warp_max = warpReduceMax(local.x);
  if (local.y != 0.f) local.y /= expf(warp_max - local.x);
  float warp_sum = warpReduceSum(local.y);
  if (lane == 0) sdata[wid] = make_float2(warp_max, warp_sum);

  // sync sdata
  __syncthreads();

  if (wid == 0) {
    auto block_max_sum = warpMergeMaxSum(sdata, threadsPerBlock / 32);
    if (lane == 0) global_buf[bid] = block_max_sum;
  }

  grid.sync();

  float2 global =
      blockMergeMaxSum(global_buf, gridDim.x, sdata, threadsPerBlock / 32);

  // compute output (grid-stride: n can exceed the number of threads)
  for (int i = gid; i < n; i += stride) {
    y[i] = expf(x[i] - global.x) / global.y;
  }
}

// x, y are device pointers; partial is device scratch with `blocks` entries.
void softmax_gpu(const float* x, float* y, float2* partial, const int n,
                 const int blocks) {
  // softmax_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(x, y, partial, n);
  void* args[] = {&x, &y, &partial, (void*)&n};
  CUDA_CHECK(cudaLaunchCooperativeKernel(softmax_kernel<BLOCK_SIZE>, blocks,
                                         BLOCK_SIZE, args, 0, 0));
}

void softmax_cpu(const float* x, float* y, const int n) {
  // pass 1: max
  float max_val = x[0];
  for (int i = 1; i < n; i++) {
    max_val = std::fmax(max_val, x[i]);
  }

  // pass 2: sum of exp(x - max); double, since a float running sum over
  // millions of small terms drops most of each addition
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += expf(x[i] - max_val);
  }

  // pass 3: normalize
  for (int i = 0; i < n; i++) {
    y[i] = expf(x[i] - max_val) / sum;
  }
}

int main() {
  // init device and cuda runtime context
  CUDA_CHECK(cudaSetDevice(0));

  float *h_x, *h_y_gpu;
  float *d_x, *d_y;
  float2* d_partial;
  std::vector<float> h_y_cpu(N);

  size_t size = N * sizeof(float);

  CUDA_CHECK(cudaMallocHost(&h_x, size));
  CUDA_CHECK(cudaMallocHost(&h_y_gpu, size));

  CUDA_CHECK(cudaMalloc(&d_x, size));
  CUDA_CHECK(cudaMalloc(&d_y, size));

  // init x; a moderate range keeps most outputs well above zero
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.f, 10.f);

  for (size_t i = 0; i < N; i++) h_x[i] = dist(rng);

  CUDA_CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

  // Calculate grid size: as many blocks as can be resident at once
  int numSMs, blocksPerSM;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSM, softmax_kernel<BLOCK_SIZE>, BLOCK_SIZE, 0));
  printf("numSMs: %d, blockPerSM: %d\n", numSMs, blocksPerSM);
  int blocks = numSMs * blocksPerSM;

  CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float2)));

  // benchmark gpu kernels
  L2Flusher l2;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> gpu_ms;
  gpu_ms.reserve(kRepeat);

  for (int i = -kWarmup; i < kRepeat; i++) {
    // evict x from l2
    l2.flush();

    CUDA_CHECK(cudaEventRecord(start));
    softmax_gpu(d_x, d_y, d_partial, N, blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    if (i >= 0) {
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      gpu_ms.push_back(ms);
    }
  }
  CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(gpu_ms.begin(), gpu_ms.end());
  float gpu_median_ms = gpu_ms[kRepeat / 2];
  printf("Median GPU time: %f ms\n", gpu_median_ms);

  std::vector<double> cpu_ms;
  cpu_ms.reserve(kRepeat);
  for (int i = -kWarmup; i < kRepeat; i++) {
    auto t0 = std::chrono::steady_clock::now();
    softmax_cpu(h_x, h_y_cpu.data(), N);
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

  // element-wise relative check; outputs here span ~1e-16..1e-7, so an
  // absolute tolerance would hide wrong tiny values
  constexpr double kRtol = 1e-4;
  size_t mismatches = 0, first_bad = N;
  double max_rel_err = 0.0, gpu_sum = 0.0;
  for (size_t i = 0; i < N; i++) {
    double g = h_y_gpu[i], c = h_y_cpu[i];
    double err = std::fabs(g - c);
    gpu_sum += g;
    if (c > 0) max_rel_err = std::fmax(max_rel_err, err / c);
    // !(a <= b) also catches NaN
    if (!(err <= kRtol * std::fabs(c))) {
      if (mismatches++ == 0) first_bad = i;
    }
  }
  printf("GPU sum of outputs: %f\n", gpu_sum);
  printf("Max relative error: %e\n", max_rel_err);

  if (mismatches > 0) {
    printf("Mismatch: %zu elements, first at %zu (GPU = %e, CPU = %e)\n",
           mismatches, first_bad, h_y_gpu[first_bad], h_y_cpu[first_bad]);
  } else {
    printf("Match: all %zu elements within tolerance\n", N);
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_partial));

  CUDA_CHECK(cudaFreeHost(h_x));
  CUDA_CHECK(cudaFreeHost(h_y_gpu));

  return 0;
}
