#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 一个CTA处理128x128的C矩阵tile，CTA内放8个warp（4行x2列）
// K维度每次处理16，A/B各使用4级环形缓冲，把global memory搬运和Tensor Core计算重叠起来
// shared memory每行额外padding 8个half，避免leading dimension为16/128时的bank conflict
constexpr int kFastBlockM = 128;
constexpr int kFastBlockN = 128;
constexpr int kFastWarps = 8;
constexpr int kFastThreads = kFastWarps * 32;
constexpr int kFastSkew = 8;
constexpr int kFastAStride = kWmmaK + kFastSkew;
constexpr int kFastBStride = kFastBlockN + kFastSkew;
// 左矩阵til缓存 128 * (16 + 8) = 3072个half = 6KB
constexpr int kFastAStageElements = kFastBlockM * kFastAStride;
constexpr int kFastAStages = 4;
// 右矩阵tile缓存 16 * (128 + 8) = 2176个half = 4.25KB
constexpr int kFastBStageElements = kWmmaK * kFastBStride;
constexpr int kFastBStages = 4;

// 一个CTA使用的shared memory总量 = 6KB + 4.25KB = 10.25KB + 四级缓存 = 41KB
constexpr int kFastSharedElements = kFastAStages * kFastAStageElements + kFastBStages * kFastBStageElements;

static_assert(kFastSharedElements * sizeof(half) == 41 * 1024); // 1个cta用掉了 41 KiB

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

// 允许正在发送异步拷贝的组数小于等于2（一次正在发出3组数据的异步拷贝，只需要第一组到了就行）
__device__ __forceinline__ void cp_async_wait_group_2() {
  asm volatile("cp.async.wait_group 2;" : : : "memory");
}
#endif

using AFragment = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;
using BFragment = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;
using CFragment = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half>;

// __launch_bounds__(kFastThreads, 2) 是 CUDA 的内核启动边界提示，用于指导编译器优化寄存器分配，以平衡 occupancy 和 寄存器用量。
// kFastThreads参数表示线程块中使用的线程数，2 表示每个 SM 上最多可以同时运行的线程块数。
__global__ __launch_bounds__(kFastThreads, 2) void fp16_gemm_outer_product_sm86_pipeline_kernel(const half* __restrict__ a,const half* __restrict__ b,half* __restrict__ c,int n,int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// A的4级缓冲占4*128*24个half，B的4级缓冲占4*16*136个half，总共41 KiB shared memory
__shared__ __align__(32) half shared_storage[kFastSharedElements];
half* const shared_a = shared_storage;
half* const shared_b = shared_storage + kFastAStages * kFastAStageElements;

const int tid = threadIdx.x;
const int warp_id = tid / 32;

// 8个warp按照下面的4x2形式划分128x128输出tile
// (0,0) (0,1)
// (1,0) (1,1)
// (2,0) (2,1)
// (3,0) (3,1)
const int warp_m = warp_id / 2;
const int warp_n = warp_id % 2;

// 当前CTA负责的C矩阵tile起始位置
const int block_row = static_cast<int>(blockIdx.y) * kFastBlockM;
const int block_col = static_cast<int>(blockIdx.x) * kFastBlockN;

// 一个warp持有2个A fragment、4个B fragment，计算2x4个16x16的C fragment
// C fragment在整个K循环中保留在寄存器里，循环结束后才写回global memory
AFragment a_fragments[2];
BFragment b_fragments[4];
CFragment accumulators[2][4];
#pragma unroll
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n = 0; tile_n < 4; ++tile_n) {
    wmma::fill_fragment(accumulators[tile_m][tile_n], __float2half(0.0f));
  }
}

// 每个stage的A和B都恰好包含256个16字节向量，因此256个线程每线程各搬一个向量
const int a_row = tid / 2;
const int a_column = (tid % 2) * 8;
const int b_row = tid / 16;
const int b_column = (tid % 16) * 8;

// 先把前3个K tile从global memory异步搬到shared memory，给四级流水线预热

// 一个线程发起2*3次异步拷贝（左右矩阵各一次，三层流水）
#pragma unroll
for (int stage = 0; stage < 3; ++stage) {
  cp_async_16(shared_a + stage * kFastAStageElements + a_row * kFastAStride + a_column, a + static_cast<size_t>(block_row + a_row) * k + stage * kWmmaK + a_column);
  cp_async_16(shared_b + stage * kFastBStageElements + b_row * kFastBStride + b_column, b + static_cast<size_t>(stage * kWmmaK + b_row) * n + block_col + b_column);
  cp_async_commit();
}
cp_async_wait_group_2();
__syncthreads();

const int k_steps = k / kWmmaK;
// 沿K维度循环：从当前stage加载fragment，同时发起step+3的异步搬运
for (int step = 0; step < k_steps; ++step) {
  half* const current_a = shared_a + (step % kFastAStages) * kFastAStageElements;
  half* const current_b = shared_b + (step % kFastBStages) * kFastBStageElements;
#pragma unroll
  // shared memory -> WMMA寄存器
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
    wmma::load_matrix_sync( a_fragments[tile_m], current_a + (warp_m * 2 + tile_m) * kWmmaM * kFastAStride, kFastAStride);
  }
#pragma unroll
  for (int tile_n = 0; tile_n < 4; ++tile_n) {
    wmma::load_matrix_sync( b_fragments[tile_n], current_b + (warp_n * 4 + tile_n) * kWmmaN, kFastBStride);
  }

  // 当前stage已经被读入寄存器，可以复用该环形槽位预取step+3
  // cp.async搬运与下面的Tensor Core MMA并行执行
  const int future_step = step + 3;
  half* const future_a = shared_a + (future_step % kFastAStages) * kFastAStageElements;
  half* const future_b = shared_b + (future_step % kFastBStages) * kFastBStageElements;
  if (future_step < k_steps) {
    // 异步搬运16bit是由每一个线程发起的，线程发起后又会进行打包管理
    cp_async_16( future_a + a_row * kFastAStride + a_column, a + static_cast<size_t>(block_row + a_row) * k + future_step * kWmmaK + a_column);
    cp_async_16( future_b + b_row * kFastBStride + b_column, b + static_cast<size_t>(future_step * kWmmaK + b_row) * n + block_col + b_column);
  } else {
    cp_async_zero_16(future_a + a_row * kFastAStride + a_column);
    cp_async_zero_16(future_b + b_row * kFastBStride + b_column);
  }
  cp_async_commit();

#pragma unroll
  // 2x4组WMMA计算，结果继续累加在寄存器中
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < 4; ++tile_n) {
      wmma::mma_sync(accumulators[tile_m][tile_n], a_fragments[tile_m], b_fragments[tile_n], accumulators[tile_m][tile_n]);
    }
  }

  // 最多保留2组尚未完成的异步拷贝，随后同步CTA再进入下一个K tile
  cp_async_wait_group_2();

  // @todo:能用同步前做自己的事情掩盖吗
  __syncthreads();
}

#pragma unroll
// K循环结束，把每个warp持有的2x4个C fragment直接写回global memory
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n = 0; tile_n < 4; ++tile_n) {
    const int output_row = block_row + (warp_m * 2 + tile_m) * kWmmaM;
    const int output_column = block_col + (warp_n * 4 + tile_n) * kWmmaN;
    wmma::store_matrix_sync( c + static_cast<size_t>(output_row) * n + output_column, accumulators[tile_m][tile_n], n, wmma::mem_row_major);
  }
}
#endif
}

}  // namespace

// Computes row-major C[M, N] = A[M, K] * B[K, N].  Inputs, output, and the
// Tensor Core accumulators are FP16.  Passing nullptr selects the default CUDA
// stream.  The selected kernel requires M/N to match its CTA tile and K to be
// a multiple of the four-stage pipeline depth.
extern "C" cudaError_t fp16_gemm_tensor_core(const half* a,const half* b,half* c,int m,int n,int k,cudaStream_t stream) {
  
// M需要是128的倍数，N需要是128的倍数，K需要是64的倍数
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0 || m % kFastBlockM != 0 || n % kFastBlockN != 0 || k % (kFastAStages * kWmmaK) != 0) {
  return cudaErrorInvalidValue;
}

const dim3 grid(n / kFastBlockN, m / kFastBlockM);
// 每个CTA使用256个线程，静态shared memory为41 KiB
fp16_gemm_outer_product_sm86_pipeline_kernel <<<grid, kFastThreads, 0, stream>>>(a, b, c, n, k);
return cudaPeekAtLastError();
}
