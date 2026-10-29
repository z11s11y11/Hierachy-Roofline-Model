#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 一个CTA处理128x192的C矩阵tile，CTA内放8个warp（4行x2列）
// 相比v3把N方向从128加宽到192，使同一份A数据参与更多MMA计算
constexpr int kFastBlockM = 128;
constexpr int kFastWarps = 8;
constexpr int kFastThreads = kFastWarps * 32;
constexpr int kFastSkew = 8;
constexpr int kFastAStride = kWmmaK + kFastSkew;

// 128 * (16 + 8) = 3072个half = 6kB 左矩阵
constexpr int kFastAStageElements = kFastBlockM * kFastAStride;
constexpr int kFastAStages = 4;
constexpr int kFastBStages = 4;

// A/B使用4级K=16环形缓冲，一个CTA需要49 KiB shared memory
// SM86每个SM可用100 KiB，因此仍然能够同时驻留2个CTA
constexpr int kWideBlockN = 192;
constexpr int kWideTilesPerWarpN = 6;
constexpr int kWideBStride = kWideBlockN + kFastSkew;

// 16 * （192 + 8）= 3200个half = 6.25kB 右矩阵
constexpr int kWideBStageElements = kWmmaK * kWideBStride;

// 一共消耗 4*（6+6.25）= 49kB shared memory per cta
constexpr int kWideSharedElements = kFastAStages * kFastAStageElements + kFastBStages * kWideBStageElements;

// 申请的shared memory空间
constexpr int kWideSharedBytes = kWideSharedElements * sizeof(half);
constexpr int kWideBVectorsPerRow = kWideBlockN / 8;
constexpr int kWideBCopyVectors = kWmmaK * kWideBVectorsPerRow;
static_assert(kWideSharedElements * sizeof(half) == 49 * 1024);
static_assert(kWideBCopyVectors == 384);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_16(void* shared_destination, const void* global_source) {
  // 将 generic 指针转换为 Shared Memory 地址空间的偏移
  const unsigned int shared_address = static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));

  // cp: 内存拷贝指令
  // async: 异步执行
  // cg: cache global只缓存到L2，避免占用L1
  // shared: 目的地是shared memory
  // global: 源是global memory
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;":: "r"(shared_address), "l"(global_source): "memory");
  // %0: 32-bit register, shared mem 地址
  // %1: 64-bit register, global mem 地址
  // shared_destination 和 global_source 都需 16 字节对齐，只拷贝 16 字节
  // "memory"，内存屏障，防止编译器将其他内存操作重排到这条指令前后
  // cp.async 是 SM80+ 指令，旧 GPU 不支持
}

__device__ __forceinline__ void cp_async_zero_16(void* shared_destination) {
  const unsigned int shared_address = static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));

  // 第 4 个参数 = 0 → zero-fill 模式
  //硬件忽略 [%1] 的实际内容
  //[%1] 只是语法上的占位符
  //传 shared_destination 或任何地址都可以，因为不会被读取
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;" : : "r"(shared_address), "l"(shared_destination) : "memory");
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
// 49 KiB超过CUDA默认的48 KiB静态上限，因此在host端设置attribute后使用动态shared memory
// 238行左右

//正常的编写模式： __shared__ __align__(32) half shared_storage[kFastSharedElements];
extern __shared__ __align__(32) half shared_storage[];

half* const shared_a = shared_storage;
half* const shared_b = shared_storage + kFastAStages * kFastAStageElements;

const int tid = threadIdx.x;
const int warp_id = tid / 32;

// 8个warp按照4x2划分输出tile；每个warp计算2x6个16x16 fragment，也就是32x96
const int warp_m = warp_id / 2;
const int warp_n = warp_id % 2;

// 当前CTA处理的128x192输出块起始位置
const int block_row = static_cast<int>(blockIdx.y) * kFastBlockM;
const int block_col = static_cast<int>(blockIdx.x) * kWideBlockN;

// A fragment复用6次B fragment，C fragment在整个K循环中驻留寄存器
AFragment a_fragments[2];

// kWideTilesPerWarpN = 6 （192/32=6）
BFragment b_fragments[kWideTilesPerWarpN];
CFragment accumulators[2][kWideTilesPerWarpN];
#pragma unroll
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    wmma::fill_fragment(accumulators[tile_m][tile_n], __float2half(0.0f));
  }
}

// 每个stage的A有2*128=256个16字节向量，B有2*192=384个；前128（384-256=128）个线程额外搬运一个B向量
const int a_row = tid / 2;
const int a_column = (tid % 2) * 8;

// 先预取前3个K tile，建立四级cp.async流水线
#pragma unroll
for (int stage = 0; stage < 3; ++stage) {
  cp_async_16(shared_a + stage * kFastAStageElements + a_row * kFastAStride + a_column, a + static_cast<size_t>(block_row + a_row) * k + stage * kWmmaK + a_column);
#pragma unroll
  // 要覆盖384个16字节向量，前128个线程搬运2个向量，后128个线程搬运1个向量
  // kWideBCopyVectors = 384
  // kWideBVectorsPerRow = 192/8=24
  for (int copy = tid; copy < kWideBCopyVectors; copy += kFastThreads) {
    const int b_row = copy / kWideBVectorsPerRow;
    const int b_column = (copy % kWideBVectorsPerRow) * 8;
    cp_async_16(shared_b + stage * kWideBStageElements + b_row * kWideBStride + b_column, b + static_cast<size_t>(stage * kWmmaK + b_row) * n + block_col + b_column);
  }
  cp_async_commit();
}
cp_async_wait_group_2();
__syncthreads();

const int k_steps = k / kWmmaK;
// 每轮消费当前stage，并把step+3异步预取到刚释放的环形槽位
for (int step = 0; step < k_steps; ++step) {
  half* const current_a = shared_a + (step % kFastAStages) * kFastAStageElements;
  half* const current_b = shared_b + (step % kFastBStages) * kWideBStageElements;
#pragma unroll
  // 当前A/B tile从shared memory加载到WMMA fragment
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
    wmma::load_matrix_sync( a_fragments[tile_m], current_a + (warp_m * 2 + tile_m) * kWmmaM * kFastAStride, kFastAStride);
  }
#pragma unroll
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    wmma::load_matrix_sync( b_fragments[tile_n], current_b + (warp_n * kWideTilesPerWarpN + tile_n) * kWmmaN, kWideBStride);
  }

  // 发起未来tile的global memory -> shared memory异步搬运
  const int future_step = step + 3;
  // 下一步异步拷贝的起始地址
  half* const future_a = shared_a + (future_step % kFastAStages) * kFastAStageElements;
  half* const future_b = shared_b + (future_step % kFastBStages) * kWideBStageElements;
  if (future_step < k_steps) {
    cp_async_16(future_a + a_row * kFastAStride + a_column, a + static_cast<size_t>(block_row + a_row) * k + future_step * kWmmaK + a_column);
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors; copy += kFastThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_16( future_b + b_row * kWideBStride + b_column, b + static_cast<size_t>(future_step * kWmmaK + b_row) * n + block_col + b_column);
    }
  } else {
    cp_async_zero_16(future_a + a_row * kFastAStride + a_column);
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors; copy += kFastThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_zero_16(future_b + b_row * kWideBStride + b_column);
    }
  }
  cp_async_commit();

#pragma unroll
  //192/16=12 
  // 一个warp完成2x6组16x16x16的Tensor Core计算
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
// K维度计算完成后，把寄存器中的C fragment直接写回global memory
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    const int output_row = block_row + (warp_m * 2 + tile_m) * kWmmaM;
    const int output_column = block_col + (warp_n * kWideTilesPerWarpN + tile_n) * kWmmaN;
    wmma::store_matrix_sync( c + static_cast<size_t>(output_row) * n + output_column, accumulators[tile_m][tile_n], n, wmma::mem_row_major);
  }
}
#endif
}

// cudaFuncSetAttribute(
//     fp16_gemm_outer_product_sm86_wide_pipeline_kernel,  // 目标 kernel
//     cudaFuncAttributeMaxDynamicSharedMemorySize,         // 属性：最大动态 shared memory
//     kWideSharedBytes                                     // 值：49 KiB (50176 bytes)
// );

cudaError_t configure_sm86_wide_pipeline_kernel() {
  // 只在第一次调用时申请49 KiB动态shared memory，并优先把L1/shared carveout给shared memory
  static const cudaError_t configuration_error = [] {
    cudaError_t error = cudaFuncSetAttribute( fp16_gemm_outer_product_sm86_wide_pipeline_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kWideSharedBytes);
    if (error != cudaSuccess) return error;
    return cudaFuncSetAttribute( fp16_gemm_outer_product_sm86_wide_pipeline_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  }();
  return configuration_error;
}

// SM 内部 SRAM 分配（A6000, 128 KB 总 SRAM）：

// Carveout = 0（默认）:   L1 多，Shared 少（如 128 KB L1 + 0 KB Shared）
// Carveout = 50:          平衡（如 64 KB L1 + 64 KB Shared）
// Carveout = 100:         L1 少，Shared 多（如 28 KB L1 + 100 KB Shared）

// 实际：
// - carveout 是"偏好"不是"保证"
// - 100 表示"尽量多给 shared memory"
// - 实际分配依架构和驱动而定



}  // namespace

// Computes row-major C[M, N] = A[M, K] * B[K, N].  Inputs, output, and the
// Tensor Core accumulators are FP16.  Passing nullptr selects the default CUDA
// stream.  The selected kernel requires M/N to match its CTA tile and K to be
// a multiple of the four-stage pipeline depth.
extern "C" cudaError_t fp16_gemm_tensor_core(const half* a,const half* b,half* c,int m,int n,int k,cudaStream_t stream) {
// 只保留128x192最优核，因此M/N/K必须分别按128/192/64对齐
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
    k <= 0 || m % kFastBlockM != 0 || n % kWideBlockN != 0 || k % (kFastAStages * kWmmaK) != 0) {
  return cudaErrorInvalidValue;
}

const cudaError_t error = configure_sm86_wide_pipeline_kernel();
if (error != cudaSuccess) return error;
const dim3 grid(n / kWideBlockN, m / kFastBlockM);
// 256线程/CTA，launch时传入49 KiB动态shared memory
fp16_gemm_outer_product_sm86_wide_pipeline_kernel
    <<<grid, kFastThreads, kWideSharedBytes, stream>>>(a, b, c, n, k);
return cudaPeekAtLastError();
}
