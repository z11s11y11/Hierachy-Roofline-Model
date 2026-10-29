#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GEMM_DEFAULT_M
#define GEMM_DEFAULT_M 8192
#endif
#ifndef GEMM_DEFAULT_N
#define GEMM_DEFAULT_N 8192
#endif
#ifndef GEMM_DEFAULT_K
#define GEMM_DEFAULT_K 8192
#endif
#ifndef GEMM_MIN_SM
#define GEMM_MIN_SM 70
#endif
#ifndef GEMM_DESCRIPTION
#define GEMM_DESCRIPTION "FP16 Tensor Core GEMM"
#endif

extern "C" cudaError_t fp16_gemm_tensor_core(const half*, const half*, half*,
                                              int, int, int, cudaStream_t);

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';     \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (false)

namespace {

int positive_int(const char* text) {
  size_t parsed = 0;
  const long long value = std::stoll(text, &parsed);
  if (text[parsed] != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("arguments must be positive integers");
  }
  return static_cast<int>(value);
}

bool validate(const std::vector<half>& a, const std::vector<half>& b,
              const std::vector<half>& c, int m, int n, int k) {
  constexpr int kSamples = 128;
  std::mt19937 rng(20260726);
  std::uniform_int_distribution<int> rows(0, m - 1);
  std::uniform_int_distribution<int> columns(0, n - 1);
  float max_error = 0.0f;

  for (int sample = 0; sample < kSamples; ++sample) {
    const int row = sample < 4 ? ((sample == 1 || sample == 3) ? m - 1 : 0)
                               : rows(rng);
    const int column = sample < 4 ? ((sample == 1 || sample == 2) ? n - 1 : 0)
                                  : columns(rng);
    float reference = 0.0f;
    float product_sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
      const float product =
          __half2float(a[static_cast<size_t>(row) * k + inner]) *
          __half2float(b[static_cast<size_t>(inner) * n + column]);
      reference += product;
      product_sum += std::abs(product);
    }
    const float actual = __half2float(c[static_cast<size_t>(row) * n + column]);
    const float error = std::abs(actual - reference);
    max_error = std::max(max_error, error);
    if (!std::isfinite(actual) || error > 0.02f + 0.015f * product_sum) {
      std::cerr << "Mismatch at (" << row << ", " << column << "): GPU="
                << actual << ", reference=" << reference << '\n';
      return false;
    }
  }
  std::cout << "Validation: PASS (" << kSamples
            << " samples, max abs error " << max_error << ")\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int m = GEMM_DEFAULT_M;
  int n = GEMM_DEFAULT_N;
  int k = GEMM_DEFAULT_K;
  int warmup = 5;
  int iterations = 20;
  try {
    if (argc != 1 && argc != 4 && argc != 6) {
      throw std::invalid_argument(
          "usage: program [M N K [warmup_iterations iterations]]");
    }
    if (argc >= 4) {
      m = positive_int(argv[1]);
      n = positive_int(argv[2]);
      k = positive_int(argv[3]);
    }
    if (argc == 6) {
      warmup = positive_int(argv[4]);
      iterations = positive_int(argv[5]);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major * 10 + properties.minor < GEMM_MIN_SM) {
    std::cerr << "This kernel requires SM " << GEMM_MIN_SM / 10 << '.'
              << GEMM_MIN_SM % 10 << " or newer\n";
    return EXIT_FAILURE;
  }

  const size_t a_count = static_cast<size_t>(m) * k;
  const size_t b_count = static_cast<size_t>(k) * n;
  const size_t c_count = static_cast<size_t>(m) * n;
  std::vector<half> host_a(a_count), host_b(b_count), host_c(c_count);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> values(-0.25f, 0.25f);
  for (half& value : host_a) value = __float2half(values(rng));
  for (half& value : host_b) value = __float2half(values(rng));

  half *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, a_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_b, b_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_c, c_count * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), a_count * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), b_count * sizeof(half),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(fp16_gemm_tensor_core(device_a, device_b, device_c, m, n, k, 0));
  }
  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, c_count * sizeof(half),
                        cudaMemcpyDeviceToHost));
  const bool correct = validate(host_a, host_b, host_c, m, n, k);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(fp16_gemm_tensor_core(device_a, device_b, device_c, m, n, k, 0));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const double milliseconds = total_ms / iterations;
  const double tflops = 2.0 * m * n * k / (milliseconds * 1.0e9);

  std::cout << "GPU: " << properties.name << "\nKernel: " << GEMM_DESCRIPTION
            << "\nShape: " << m << " x " << n << " x " << k
            << "\nTime: " << std::fixed << std::setprecision(3) << milliseconds
            << " ms\nPerformance: " << tflops << " TFLOPS\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
