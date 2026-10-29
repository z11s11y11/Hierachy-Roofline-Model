// Supplemental cache/DRAM and dense FP16 Tensor Core calibration.
// ERT's read/write kernel does not establish an L1 load bandwidth ceiling.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)

template<bool L1>
__device__ __forceinline__ float4 load(const float4* p) {
  float4 v;
  if constexpr (L1) {
    asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
  } else {
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
  }
  return v;
}

template<bool L1>
__global__ void read_bandwidth(const float4* input, float4* output, size_t count, int repeats) {
  float4 sum = make_float4(0, 0, 0, 0);
  // The L1 probe gives each CTA its own 4 KiB, repeatedly loaded with .ca.
  // L2/DRAM probes use .cg to bypass L1 and sweep their complete working set.
  const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int r = 0; r < repeats; ++r) {
    for (size_t i = first; i < count; i += size_t(gridDim.x) * blockDim.x) {
      float4 v = load<L1>(input + i);
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
  }
  output[first] = sum;
}

__global__ void tensor_peak(half* output, int repeats) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c[8];
  wmma::fill_fragment(a, __float2half(0.0078125f));
  wmma::fill_fragment(b, __float2half(0.0078125f));
  #pragma unroll
  for (int j = 0; j < 8; ++j) wmma::fill_fragment(c[j], __float2half(0.f));
  #pragma unroll 1
  for (int r = 0; r < repeats; ++r) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) wmma::mma_sync(c[j], a, b, c[j]);
  }
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  #pragma unroll
  for (int j = 0; j < 8; ++j)
    wmma::store_matrix_sync(output + (size_t(warp) * 8 + j) * 256, c[j], 16,
                            wmma::mem_row_major);
}

int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr, "usage: calibrate info|L1|L2|DRAM|tensor\n"); return 1; }
  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, 0));
  std::string mode = argv[1];
  if (mode == "info") {
    printf("{\"name\":\"%s\",\"sm\":%d,\"sms\":%d,\"memory_bytes\":%zu,"
           "\"l2_bytes\":%d,\"memory_bus_bits\":%d,\"memory_clock_khz\":%d}\n",
           p.name, p.major * 10 + p.minor, p.multiProcessorCount,
           p.totalGlobalMem, p.l2CacheSize, p.memoryBusWidth, p.memoryClockRate);
    return 0;
  }
  const int blocks = p.multiProcessorCount * 4, threads = 256;
  float4 *input = nullptr, *output = nullptr;
  half* tensor_output = nullptr;
  size_t count = 0;
  int repeats = 0;
  if (mode == "tensor") {
    // Keep the exactly representable result below 1.0: FP16 accumulation
    // otherwise rounds away the half-step HMMA updates and saturates at 1.
    repeats = 1000;
    CHECK(cudaMalloc(&tensor_output, size_t(blocks) * (threads / 32) * 8 * 256 * sizeof(half)));
  } else {
    if (mode == "L1") { count = size_t(blocks) * threads; repeats = 1024; }
    else if (mode == "L2") { count = size_t(p.l2CacheSize) / 2 / sizeof(float4); repeats = 128; }
    else if (mode == "DRAM") {
      count = std::max(size_t(64) << 20, size_t(p.l2CacheSize) * 16) / sizeof(float4);
      repeats = 8;
    } else { fprintf(stderr, "unknown mode\n"); return 1; }
    // At least one whole grid is needed for validation and coalescing.
    if (count < size_t(blocks) * threads) { fprintf(stderr, "L2 working set too small\n"); return 1; }
    CHECK(cudaMalloc(&input, count * sizeof(float4)));
    CHECK(cudaMalloc(&output, size_t(blocks) * threads * sizeof(float4)));
    std::vector<float4> host(count, make_float4(1, 1, 1, 1));
    CHECK(cudaMemcpy(input, host.data(), count * sizeof(float4), cudaMemcpyHostToDevice));
  }
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  auto launch = [&] {
    if (mode == "tensor") tensor_peak<<<blocks, threads>>>(tensor_output, repeats);
    else if (mode == "L1") read_bandwidth<true><<<blocks, threads>>>(input, output, count, repeats);
    else read_bandwidth<false><<<blocks, threads>>>(input, output, count, repeats);
    CHECK(cudaGetLastError());
  };
  launch(); CHECK(cudaDeviceSynchronize());  // ncu skips this warmup
  CHECK(cudaEventRecord(start));
  launch();
  CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
  float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
  if (mode == "tensor") {
    std::vector<half> host(size_t(blocks) * (threads / 32) * 8 * 256);
    CHECK(cudaMemcpy(host.data(), tensor_output, host.size() * sizeof(half), cudaMemcpyDeviceToHost));
    for (half v : host) if (fabs(__half2float(v) - repeats / 1024.f) > 0.01f) {
      fprintf(stderr, "Tensor validation failed: actual=%g expected=%g\n",
              __half2float(v), repeats / 1024.f); return 1;
    }
  } else {
    std::vector<float4> host(size_t(blocks) * threads);
    CHECK(cudaMemcpy(host.data(), output, host.size() * sizeof(float4), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
      float expected = i < count ? ((count - 1 - i) / host.size() + 1) * repeats : 0;
      float4 v = host[i];
      if (v.x != expected || v.y != expected || v.z != expected || v.w != expected) {
        fprintf(stderr, "Read validation failed at %zu\n", i); return 1;
      }
    }
  }
  const double flops = mode == "tensor" ? double(blocks) * (threads / 32) * 8 * repeats * 8192 : 0;
  printf("{\"mode\":\"%s\",\"working_set_bytes\":%zu,\"repeats\":%d,"
         "\"logical_bytes\":%.0f,\"flops\":%.0f,\"event_ms\":%.9g,\"validation\":\"PASS\"}\n",
         mode.c_str(), count * sizeof(float4), repeats, double(count) * sizeof(float4) * repeats, flops, ms);
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  if (input) CHECK(cudaFree(input));
  if (output) CHECK(cudaFree(output));
  if (tensor_output) CHECK(cudaFree(tensor_output));
}
