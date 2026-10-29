#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 把连续的物理CTA按4个M方向tile分组：先沿M走完一组，再移动到下一个N tile
// 相邻CTA能够复用同一个B panel的L2 cache，GROUP_M=4是A6000实测最优配置
constexpr int kWideGroupedCtaM = 4;

// 这是一个 Grouped CTA Swizzle 函数，用于重新排列 GPU 线程块（CTA/Block）的执行顺序，以改善 L2 Cache 的数据局部性


// 原始 Row-Major 调度（kGroupM = 1）：
// CTA(0,0) → CTA(0,1) → CTA(0,2) → CTA(0,3) → CTA(1,0) → ...

// 问题：
// - CTA(0,0) 和 CTA(1,0) 共享 B[0:K][0:N_tile]（B 的同一列 panel）
// - 但它们在调度上相隔很远（中间隔了整行）
// - CTA(1,0) 执行时，B panel 可能已被踢出 L2

// Grouped 调度（kGroupM = 4）：
// Group 0: CTA(0,0) → CTA(1,0) → CTA(2,0) → CTA(3,0) → CTA(0,1) → CTA(1,1) ...
//          ↑         ↑         ↑         ↑
//          共享 B[0]  共享 B[0]  共享 B[0]  共享 B[0]

// → 同组 CTA 连续访问相同的 B panel
// → B 数据在 L2 中命中率高
// → 减少 HBM 带宽压力

template <int kGroupM>
__device__ __forceinline__ void grouped_cta_tile_coordinates( int* tile_m, int* tile_n) {
  // M N 方向总 tile 数
  const int tiles_m = static_cast<int>(gridDim.y);
  const int tiles_n = static_cast<int>(gridDim.x);
  // 当前 CTA 的物理序号
  const int physical_tile = static_cast<int>(blockIdx.y) * tiles_n + static_cast<int>(blockIdx.x);

  if constexpr (kGroupM == 1) {
    // 直接使用原始坐标
    *tile_m = static_cast<int>(blockIdx.y);
    *tile_n = static_cast<int>(blockIdx.x);
  } else {
    const int tiles_per_group = kGroupM * tiles_n;  // 每组包含的 tile 数
    const int group_id = physical_tile / tiles_per_group;  // 当前属于第几组
    const int first_tile_m = group_id * kGroupM;  // 组起始 M 位置
    const int actual_group_m = min(kGroupM, tiles_m - first_tile_m);  // 边界处理
    const int tile_in_group = physical_tile - group_id * tiles_per_group;  // 组内偏移

    // 组内M变化最快，使相邻CTA访问相同的B panel；一组M完成后N再向前移动
    *tile_m = first_tile_m + tile_in_group % actual_group_m;
    *tile_n = tile_in_group / actual_group_m;
  }
}

// 一个CTA处理128x192的C矩阵tile，CTA内放8个warp（4行x2列）
constexpr int kFastBlockM = 128;
constexpr int kFastWarps = 8;
constexpr int kFastThreads = kFastWarps * 32;
constexpr int kFastSkew = 8;
constexpr int kFastAStride = kWmmaK + kFastSkew;
constexpr int kFastAStageElements = kFastBlockM * kFastAStride;
constexpr int kFastAStages = 4;
constexpr int kFastBStages = 4;

// A/B使用4级K=16环形缓冲，一个CTA使用49 KiB shared memory，SM86仍可驻留2个CTA
constexpr int kWideBlockN = 192;
constexpr int kWideTilesPerWarpN = 6;
constexpr int kWideBStride = kWideBlockN + kFastSkew;
constexpr int kWideBStageElements = kWmmaK * kWideBStride;
constexpr int kWideSharedElements =
    kFastAStages * kFastAStageElements +
    kFastBStages * kWideBStageElements;
constexpr int kWideSharedBytes = kWideSharedElements * sizeof(half);
constexpr int kWideBVectorsPerRow = kWideBlockN / 8;
constexpr int kWideBCopyVectors = kWmmaK * kWideBVectorsPerRow;
static_assert(kWideSharedElements * sizeof(half) == 49 * 1024);
static_assert(kWideBCopyVectors == 384);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16(void* shared_destination,
                                            const void* global_source) {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(global_source)
               : "memory");
}

__device__ __forceinline__ void cp_async_zero_16(void* shared_destination) {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;"
               :
               : "r"(shared_address), "l"(shared_destination)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" : : : "memory");
}

// 允许异步拷贝的次数小于等于2
__device__ __forceinline__ void cp_async_wait_group_2() {
  asm volatile("cp.async.wait_group 2;" : : : "memory");
}
#endif

using AFragment = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;
using BFragment = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;
using CFragment = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half>;

__global__ __launch_bounds__(kFastThreads, 2) void fp16_gemm_outer_product_sm86_wide_pipeline_kernel(const half* __restrict__ a,const half* __restrict__ b,half* __restrict__ c,int n,int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// 49 KiB超过默认48 KiB上限，host端设置attribute后通过动态shared memory传入
extern __shared__ __align__(32) half shared_storage[];
half* const shared_a = shared_storage;
half* const shared_b =
    shared_storage + kFastAStages * kFastAStageElements;

const int tid = threadIdx.x;
const int warp_id = tid / 32;

// 8个warp按4x2排列，每个warp计算2x6个16x16的C fragment
const int warp_m = warp_id / 2;
const int warp_n = warp_id % 2;

// 逻辑tile坐标使用GROUP_M=4映射，而不是直接采用blockIdx.x/y
int tile_m = 0;
int tile_n = 0;
grouped_cta_tile_coordinates<kWideGroupedCtaM>(&tile_m, &tile_n);
const int block_row = tile_m * kFastBlockM;
const int block_col = tile_n * kWideBlockN;

// A/B fragment负责本轮K tile，C fragment在完整K循环中一直驻留寄存器
AFragment a_fragments[2];
BFragment b_fragments[kWideTilesPerWarpN];
CFragment accumulators[2][kWideTilesPerWarpN];
#pragma unroll
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    wmma::fill_fragment(accumulators[tile_m][tile_n],
                        __float2half(0.0f));
  }
}

// 每个stage的A有256个16字节向量，B有384个；前128个线程多搬一个B向量
const int a_row = tid / 2;
const int a_column = (tid % 2) * 8;

#pragma unroll
// 预取前3个K tile，为四级cp.async环形流水线预热
for (int stage = 0; stage < 3; ++stage) {
  cp_async_16(shared_a + stage * kFastAStageElements +
                  a_row * kFastAStride + a_column,
              a + static_cast<size_t>(block_row + a_row) * k +
                  stage * kWmmaK + a_column);
#pragma unroll
  for (int copy = tid; copy < kWideBCopyVectors;
       copy += kFastThreads) {
    const int b_row = copy / kWideBVectorsPerRow;
    const int b_column = (copy % kWideBVectorsPerRow) * 8;
    cp_async_16(shared_b + stage * kWideBStageElements +
                    b_row * kWideBStride + b_column,
                b + static_cast<size_t>(stage * kWmmaK + b_row) * n +
                    block_col + b_column);
  }
  cp_async_commit();
}
cp_async_wait_group_2();
__syncthreads();

const int k_steps = k / kWmmaK;
// 消费当前stage，同时把step+3预取到已经释放的stage
for (int step = 0; step < k_steps; ++step) {
  half* const current_a =
      shared_a + (step % kFastAStages) * kFastAStageElements;
  half* const current_b =
      shared_b + (step % kFastBStages) * kWideBStageElements;
#pragma unroll
  // shared memory -> WMMA寄存器
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
    wmma::load_matrix_sync( a_fragments[tile_m], current_a + (warp_m * 2 + tile_m) * kWmmaM * kFastAStride, kFastAStride);
  }
#pragma unroll
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    wmma::load_matrix_sync( b_fragments[tile_n], current_b + (warp_n * kWideTilesPerWarpN + tile_n) * kWmmaN, kWideBStride);
  }

  // global memory -> shared memory，和当前tile的MMA重叠执行
  const int future_step = step + 3;
  half* const future_a =
      shared_a + (future_step % kFastAStages) * kFastAStageElements;
  half* const future_b =
      shared_b + (future_step % kFastBStages) * kWideBStageElements;
  if (future_step < k_steps) {
    cp_async_16(
        future_a + a_row * kFastAStride + a_column,
        a + static_cast<size_t>(block_row + a_row) * k +
            future_step * kWmmaK + a_column);
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors;
         copy += kFastThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_16(
          future_b + b_row * kWideBStride + b_column,
          b + static_cast<size_t>(future_step * kWmmaK + b_row) * n +
              block_col + b_column);
    }
  } else {
    cp_async_zero_16(future_a + a_row * kFastAStride + a_column);
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors;
         copy += kFastThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_zero_16(future_b + b_row * kWideBStride + b_column);
    }
  }
  cp_async_commit();

#pragma unroll
  // 一个warp完成2x6组16x16x16 Tensor Core计算
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
      wmma::mma_sync(accumulators[tile_m][tile_n], a_fragments[tile_m], b_fragments[tile_n], accumulators[tile_m][tile_n]);
    }
  }

  cp_async_wait_group_2();
  __syncthreads();
}

#pragma unroll
// K维度结束，把寄存器中的C fragment写回global memory
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    const int output_row =
        block_row + (warp_m * 2 + tile_m) * kWmmaM;
    const int output_column =
        block_col +
        (warp_n * kWideTilesPerWarpN + tile_n) * kWmmaN;
    wmma::store_matrix_sync(
        c + static_cast<size_t>(output_row) * n + output_column,
        accumulators[tile_m][tile_n], n, wmma::mem_row_major);
  }
}
#endif
}

cudaError_t configure_sm86_wide_pipeline_kernel() {
  // 第一次调用时申请49 KiB动态shared memory，并选择100%的shared memory carveout
  static const cudaError_t configuration_error = [] {
    cudaError_t error = cudaFuncSetAttribute(
        fp16_gemm_outer_product_sm86_wide_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kWideSharedBytes);
    if (error != cudaSuccess) return error;
    return cudaFuncSetAttribute(
        fp16_gemm_outer_product_sm86_wide_pipeline_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  }();
  return configuration_error;
}

}  // namespace

// Computes row-major C[M, N] = A[M, K] * B[K, N].  Inputs, output, and the
// Tensor Core accumulators are FP16.  Passing nullptr selects the default CUDA
// stream.  The selected kernel requires M/N to match its CTA tile and K to be
// a multiple of the four-stage pipeline depth.
extern "C" cudaError_t fp16_gemm_tensor_core(const half* a,const half* b,half* c,int m,int n,int k,cudaStream_t stream) {
// 只保留128x192、GROUP_M=4的最优核，尺寸必须按128/192/64对齐
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
    k <= 0 || m % kFastBlockM != 0 || n % kWideBlockN != 0 ||
    k % (kFastAStages * kWmmaK) != 0) {
  return cudaErrorInvalidValue;
}

const cudaError_t error = configure_sm86_wide_pipeline_kernel();
if (error != cudaSuccess) return error;
const dim3 grid(n / kWideBlockN, m / kFastBlockM);
fp16_gemm_outer_product_sm86_wide_pipeline_kernel
    <<<grid, kFastThreads, kWideSharedBytes, stream>>>(a, b, c, n, k);
return cudaPeekAtLastError();
}
