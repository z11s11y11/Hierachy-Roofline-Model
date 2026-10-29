// One common correctness/timing/profile harness for all ten implementations.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" cudaError_t fp16_gemm_tensor_core(const half*, const half*, half*,
                                             int, int, int, cudaStream_t);
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc != 6) { fprintf(stderr, "usage: runner M N K warmups iterations\n"); return 1; }
  int values[5];
  try {
    for (int i = 0; i < 5; ++i) {
      size_t end = 0;
      long long value = std::stoll(argv[i + 1], &end);
      if (argv[i + 1][end] || value <= 0 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument("positive integers required");
      values[i] = int(value);
    }
  } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
  int m = values[0], n = values[1], k = values[2], warmups = values[3], iterations = values[4];
  size_t as = size_t(m) * k, bs = size_t(k) * n, cs = size_t(m) * n;
  std::vector<half> a(as), b(bs), c(cs);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  for (half& v : a) v = __float2half(distribution(rng));
  for (half& v : b) v = __float2half(distribution(rng));
  half *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(cudaMalloc(&da, as * sizeof(half)));
  CHECK(cudaMalloc(&db, bs * sizeof(half)));
  CHECK(cudaMalloc(&dc, cs * sizeof(half)));
  CHECK(cudaMemcpy(da, a.data(), as * sizeof(half), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b.data(), bs * sizeof(half), cudaMemcpyHostToDevice));
  for (int i = 0; i < warmups; ++i) CHECK(fp16_gemm_tensor_core(da, db, dc, m, n, k, 0));
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) CHECK(fp16_gemm_tensor_core(da, db, dc, m, n, k, 0));
  CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
  float total_ms;
  CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  // Push/pop NVTX syntax lets ncu collect every kernel of this one invocation.
  nvtxRangePushA("roofline");
  CHECK(fp16_gemm_tensor_core(da, db, dc, m, n, k, 0));
  CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
  CHECK(cudaMemcpy(c.data(), dc, cs * sizeof(half), cudaMemcpyDeviceToHost));

  std::mt19937 samples(20261029);
  std::uniform_int_distribution<int> row_dist(0, m - 1), col_dist(0, n - 1);
  double error_squared = 0, reference_squared = 0, max_error = 0;
  for (int sample = 0; sample < 128; ++sample) {
    int row = sample < 4 ? (sample % 2 ? m - 1 : 0) : row_dist(samples);
    int col = sample < 4 ? (sample / 2 ? n - 1 : 0) : col_dist(samples);
    double expected = 0;
    for (int j = 0; j < k; ++j)
      expected += double(__half2float(a[size_t(row) * k + j])) * __half2float(b[size_t(j) * n + col]);
    double actual = __half2float(c[size_t(row) * n + col]);
    double error = std::abs(actual - expected);
    if (!std::isfinite(actual) || error > 0.02 + 0.03 * std::abs(expected)) {
      fprintf(stderr, "Validation failed (%d,%d): actual=%g expected=%g error=%g\n", row, col, actual, expected, error);
      return 1;
    }
    max_error = std::max(max_error, error);
    error_squared += error * error;
    reference_squared += expected * expected;
  }
  double relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
  if (relative_l2 > 0.03) { fprintf(stderr, "Validation relative L2 error %g > 0.03\n", relative_l2); return 1; }
  double ms = total_ms / iterations;
  printf("{\"event_ms\":%.9g,\"TFLOPS\":%.9g,\"validation\":\"PASS\","
         "\"validation_samples\":128,\"max_abs_error\":%.9g,\"relative_l2_error\":%.9g}\n",
         ms, 2.0 * m * n * k / ms / 1e9, max_error, relative_l2);
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
}
