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

// Debugging: the LeetGPU example (3x3 input, 2x2 kernel -> 2x2 output), run
// once. Benchmark: kUseExample false, kWarmup 3, kRepeat 20.
constexpr bool kUseExample = false;
constexpr int kWarmup = 3;
constexpr int kRepeat = 20;
// The CPU reference is slow at full size: fixed, fewer runs.
constexpr int kCpuWarmup = 1;
constexpr int kCpuRepeat = 3;

// Random problem size used when kUseExample is false (LeetGPU limits: input
// up to 3072x3072, kernel up to 31x31).
constexpr int kInputRows = 3072;
constexpr int kInputCols = 3072;
constexpr int kKernelRows = 31;
constexpr int kKernelCols = 31;

__constant__ float c_kernel[31 * 31];

// ---------------------------- LeetGPU solution ----------------------------
__global__ void conv_2d_kernel(const float* input, const float* kernel,
                               float* output, int input_rows, int input_cols,
                               int kernel_rows, int kernel_cols) {
  int o_row = blockIdx.y * blockDim.y + threadIdx.y;
  int o_col = blockIdx.x * blockDim.x + threadIdx.x;

  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;

  if (o_row < output_rows && o_col < output_cols) {
    float s = 0.f;
    for (int i = 0; i < kernel_rows; i++) {
      for (int j = 0; j < kernel_cols; j++) {
        int i_row = o_row + i;
        int i_col = o_col + j;

        if (i_row < input_rows && i_col < input_cols) {
          s +=
              input[i_row * input_cols + i_col] * c_kernel[i * kernel_cols + j];
        }
      }
    }
    output[o_row * output_cols + o_col] = s;
  }
}

void conv_2d(const float* input, const float* kernel, float* output,
             int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;

  dim3 threads(16, 16);
  dim3 blocks((output_cols + 15) / 16, (output_rows + 15) / 16);

  CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, kernel,
                                sizeof(float) * kernel_cols * kernel_rows, 0,
                                cudaMemcpyDeviceToDevice));

  conv_2d_kernel<<<blocks, threads>>>(input, kernel, output, input_rows,
                                      input_cols, kernel_rows, kernel_cols);
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  conv_2d(input, kernel, output, input_rows, input_cols, kernel_rows,
          kernel_cols);
  cudaDeviceSynchronize();
}

// -------------------------- end LeetGPU solution --------------------------

// Valid 2D cross-correlation (no kernel flip), as LeetGPU defines it:
// out[r][c] = sum_{i,j} in[r + i][c + j] * k[i][j]
void conv_2d_cpu(const float* input, const float* kernel, float* output,
                 int input_rows, int input_cols, int kernel_rows,
                 int kernel_cols) {
  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;
  for (int r = 0; r < output_rows; r++) {
    for (int c = 0; c < output_cols; c++) {
      double sum = 0.0;
      for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0; j < kernel_cols; j++) {
          sum += static_cast<double>(input[(r + i) * input_cols + c + j]) *
                 kernel[i * kernel_cols + j];
        }
      }
      output[r * output_cols + c] = static_cast<float>(sum);
    }
  }
}

void print_matrix(const char* name, const float* m, int rows, int cols) {
  printf("%s (%dx%d):\n", name, rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) printf(" %8.3f", m[r * cols + c]);
    printf("\n");
  }
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  int input_rows = kInputRows, input_cols = kInputCols;
  int kernel_rows = kKernelRows, kernel_cols = kKernelCols;
  std::vector<float> h_input, h_kernel;

  if (kUseExample) {
    // LeetGPU example 1; expected output [[6, 8], [12, 14]]
    input_rows = input_cols = 3;
    kernel_rows = kernel_cols = 2;
    h_input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    h_kernel = {0, 1, 1, 0};
  } else {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    h_input.resize(static_cast<size_t>(input_rows) * input_cols);
    h_kernel.resize(static_cast<size_t>(kernel_rows) * kernel_cols);
    for (float& v : h_input) v = dist(rng);
    for (float& v : h_kernel) v = dist(rng);
  }

  int output_rows = input_rows - kernel_rows + 1;
  int output_cols = input_cols - kernel_cols + 1;
  size_t n_out = static_cast<size_t>(output_rows) * output_cols;
  printf("input %dx%d, kernel %dx%d, output %dx%d\n", input_rows, input_cols,
         kernel_rows, kernel_cols, output_rows, output_cols);

  std::vector<float> h_output_gpu(n_out), h_output_cpu(n_out);

  float *d_input, *d_kernel, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, h_kernel.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n_out * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(),
                        h_kernel.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // benchmark gpu kernels
  L2Flusher l2;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> gpu_ms;
  gpu_ms.reserve(kRepeat);

  for (int i = -kWarmup; i < kRepeat; i++) {
    // poison the output so a kernel that skips elements cannot pass on stale
    // values from the previous run
    CUDA_CHECK(cudaMemset(d_output, 0xff, n_out * sizeof(float)));
    l2.flush();

    CUDA_CHECK(cudaEventRecord(start));
    solve(d_input, d_kernel, d_output, input_rows, input_cols, kernel_rows,
          kernel_cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    if (i >= 0) {
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      gpu_ms.push_back(ms);
    }
  }
  CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, n_out * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(gpu_ms.begin(), gpu_ms.end());
  float gpu_median_ms = gpu_ms[kRepeat / 2];
  printf("Median GPU time: %f ms\n", gpu_median_ms);

  std::vector<double> cpu_ms;
  cpu_ms.reserve(kCpuRepeat);
  for (int i = -kCpuWarmup; i < kCpuRepeat; i++) {
    auto t0 = std::chrono::steady_clock::now();
    conv_2d_cpu(h_input.data(), h_kernel.data(), h_output_cpu.data(),
                input_rows, input_cols, kernel_rows, kernel_cols);
    auto t1 = std::chrono::steady_clock::now();

    if (i >= 0) {
      cpu_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(cpu_ms.begin(), cpu_ms.end());
  double cpu_median_ms = cpu_ms[kCpuRepeat / 2];
  printf("Median CPU time: %f ms\n", cpu_median_ms);
  printf("Speedup: %f\n", cpu_median_ms / gpu_median_ms);

  if (n_out <= 64) {
    print_matrix("GPU output", h_output_gpu.data(), output_rows, output_cols);
    print_matrix("CPU output", h_output_cpu.data(), output_rows, output_cols);
  }

  // element-wise |g - c| <= atol + rtol * |c|
  constexpr double kAtol = 1e-4, kRtol = 1e-4;
  size_t mismatches = 0, first_bad = n_out;
  double max_abs_err = 0.0;
  for (size_t i = 0; i < n_out; i++) {
    double g = h_output_gpu[i], c = h_output_cpu[i];
    double err = std::fabs(g - c);
    max_abs_err = std::fmax(max_abs_err, err);
    // !(a <= b) also catches NaN
    if (!(err <= kAtol + kRtol * std::fabs(c))) {
      if (mismatches++ == 0) first_bad = i;
    }
  }
  printf("Max absolute error: %e\n", max_abs_err);

  if (mismatches > 0) {
    printf("Mismatch: %zu elements, first at (%zu, %zu) (GPU = %f, CPU = %f)\n",
           mismatches, first_bad / output_cols, first_bad % output_cols,
           h_output_gpu[first_bad], h_output_cpu[first_bad]);
  } else {
    printf("Match: all %zu elements within tolerance\n", n_out);
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_output));

  return mismatches > 0 ? 1 : 0;
}
