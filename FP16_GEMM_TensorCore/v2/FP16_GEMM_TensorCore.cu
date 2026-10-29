#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 一个CTA处理64x64的C矩阵tile，CTA内放8个warp，共256个线程
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;

// 每个stage保存A的64x16=1024 tile和B的16x64=1024 tile，共2048个half（4 KiB）
// 使用两个stage做double buffer，总shared memory为8 KiB
constexpr int kAStageElements = kBlockM * kWmmaK;
constexpr int kBStageElements = kWmmaK * kBlockN;
constexpr int kStageElements = kAStageElements + kBStageElements;
constexpr int kPipelineStages = 2;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// SM80及以上使用16字节cp.async，把global memory直接异步搬到shared memory

// 强制内联展开，消除函数调用开销
__device__ __forceinline__ void cp_async_16(void* shared_destination,const void* global_source) {
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

__device__ __forceinline__ void cp_async_commit() {

  // 将当前所有未提交的 cp.async 指令打包成一个组（group），形成一个同步边界。
  asm volatile("cp.async.commit_group;" : : : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {

  // 阻塞当前线程，直到所有已提交的 cp.async 组全部完成。
  asm volatile("cp.async.wait_all;" : : : "memory");
  // 等价于cp_async_wait_group(0);   等待未完成的组数 = 0，即全部完成
}
#endif


__global__ void fp16_gemm_tensor_core_kernel(const half* __restrict__ a,const half* __restrict__ b,half* __restrict__ c,int m,int n,int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

// 左右矩阵在两极流水的情况下需要的shared memory大小为2*2048=4096个half=8 KiB
__shared__ __align__(16) half shared_storage[kPipelineStages * kStageElements];

const int tid = threadIdx.x;
const int warp_id = tid / 32;

// 8个warp按4x2排列，每个位置负责一个16x32区域
// (0,0) (0,1)
// (1,0) (1,1)
// (2,0) (2,1)
// (3,0) (3,1)
const int warp_m = warp_id / 2;
const int warp_n_group = warp_id % 2;

// 当前CTA处理的64x64输出块起始位置
const int block_row = blockIdx.y * kBlockM;
const int block_col = blockIdx.x * kBlockN;

// 每个warp加载1个A fragment(左矩阵共1*4tile)和2个B fragment(右矩阵2*2个tile)，累加2个16x16 C fragment
// C fragment在整个K循环中一直保留在寄存器里
wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,wmma::row_major> a_fragment;
wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,wmma::row_major> b_fragment_0;
wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,wmma::row_major> b_fragment_1;
wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> accumulator_0;
wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> accumulator_1;
wmma::fill_fragment(accumulator_0, __float2half(0.0f));
wmma::fill_fragment(accumulator_1, __float2half(0.0f));

// k和n是8个half的倍数时，每个global memory行首都满足16字节对齐
// 完整tile使用cp.async；边界tile或SM80以下设备使用带越界判断的同步搬运
#if __CUDA_ARCH__ >= 800
const bool aligned_problem = (k % 8 == 0) && (n % 8 == 0);
#endif

// 先准备第0个K tile；current_stage在0和1之间切换
int current_stage = 0;
int k_base = 0;
half* initial_a = shared_storage;
half* initial_b = initial_a + kAStageElements;

#if __CUDA_ARCH__ >= 800
// 左矩阵64*16,右矩阵16*64，1024个half,1个half=16bit=2Byte
// 1024个half=128个16字节向量，128个线程搬运完毕
// 完整的A/B tile各有128个16字节向量：前128个线程搬A，后128个线程搬B
const bool initial_full = (aligned_problem) && (block_row + kBlockM <= m) && (block_col + kBlockN <= n) && (kWmmaK <= k);
if (initial_full) {
  // 一共8个warp=256个线程，前128个线程搬运A，后128个线程搬运B
  if (tid < 128) {
    // 左矩阵缓存是64*16
    // (0,0) (0,1)
    // (1,0) (1,1)
    // ...
    // (63,0) (63,1)
    const int row = tid / 2;
    const int column = (tid % 2) * 8;
    // 一个线程搬运16字节数据，1half=2Byte，16Byte=8half
    cp_async_16(initial_a + row * kWmmaK + column,a + static_cast<size_t>(block_row + row) * k + column);
  } else {
    const int chunk = tid - 128;
    //右矩阵是16*64，一行有64个half，即8组16字节向量，8个线程搬运完一行 
    const int row = chunk / 8;
    const int column = (chunk % 8) * 8;
    cp_async_16(initial_b + row * kBlockN + column,b + static_cast<size_t>(row) * n + block_col + column);
  }
  cp_async_commit();
  cp_async_wait_all();
} else
#endif
{
// 边界tile逐元素搬运并补0，这条路径也兼容SM70/SM75
  for (int index = tid; index < kAStageElements; index += kThreadsPerBlock) {
    const int local_m = index / kWmmaK;
    const int local_k = index % kWmmaK;
    const int global_m = block_row + local_m;
    initial_a[index] =(global_m < m && local_k < k)? a[static_cast<size_t>(global_m) * k + local_k]: __float2half(0.0f);
    }
  // B在shared memory中保持row-major，使warp lane读取连续的global地址
  for (int index = tid; index < kBStageElements;
       index += kThreadsPerBlock) {
    const int local_k = index / kBlockN;
    const int local_n = index % kBlockN;
    const int global_n = block_col + local_n;
    initial_b[index] =(local_k < k && global_n < n)? b[static_cast<size_t>(local_k) * n + global_n]: __float2half(0.0f);
  }
}
// __syncthreads() 是 cta 级别的同步。
__syncthreads();

// 沿K维度每次处理16：计算current_stage之前，先向另一个stage预取下一个K tile
for (; k_base < k; k_base += kWmmaK) {
  const int next_k_base = k_base + kWmmaK;
  const bool has_next = next_k_base < k;
  const int next_stage = current_stage ^ 1;

  #if __CUDA_ARCH__ >= 800
    bool next_is_async = false;
  #endif

  // double buffer：next_stage始终是另一个shared memory槽位
  if (has_next) {
    half* next_a = shared_storage + next_stage * kStageElements;
    half* next_b = next_a + kAStageElements;
#if __CUDA_ARCH__ >= 800
    const bool next_full = aligned_problem && block_row + kBlockM <= m && block_col + kBlockN <= n && next_k_base + kWmmaK <= k;
    // 完整tile用cp.async发起异步预取，不在这里等待
    if (next_full) {
      if (tid < 128) {
        const int row = tid / 2;
        const int column = (tid % 2) * 8;
        // 写入的时候可能存在bank conflict????还没想明白
        cp_async_16(next_a + row * kWmmaK + column,a + static_cast<size_t>(block_row + row) * k + next_k_base + column);
      } else {
        const int chunk = tid - 128;
        const int row = chunk / 8;
        const int column = (chunk % 8) * 8;
        cp_async_16(next_b + row * kBlockN + column,b + static_cast<size_t>(next_k_base + row) * n + block_col + column);
      }
      cp_async_commit();
      next_is_async = true;
    } else
#endif
    {
      // 边界tile同步搬运到next_stage并补0
      for (int index = tid; index < kAStageElements;
        index += kThreadsPerBlock) {
        const int local_m = index / kWmmaK;
        const int local_k = index % kWmmaK;
        const int global_m = block_row + local_m;
        const int global_k = next_k_base + local_k;
        next_a[index] =(global_m < m && global_k < k)? a[static_cast<size_t>(global_m) * k + global_k] : __float2half(0.0f);
      }
      for (int index = tid; index < kBStageElements;
           index += kThreadsPerBlock) {
        const int local_k = index / kBlockN;
        const int local_n = index % kBlockN;
        const int global_k = next_k_base + local_k;
        const int global_n = block_col + local_n;
        next_b[index] =(global_k < k && global_n < n)? b[static_cast<size_t>(global_k) * n + global_n]: __float2half(0.0f);
      }
      __syncthreads();
    }
  }

  half* current_a = shared_storage + current_stage * kStageElements;
  half* current_b = current_a + kAStageElements;
  const int warp_column_0 = warp_n_group * 2 * kWmmaN;
  const int warp_column_1 = warp_column_0 + kWmmaN;

  // shared memory -> WMMA寄存器；同一个A fragment分别和两个B fragment计算
  wmma::load_matrix_sync(a_fragment, current_a + warp_m * kWmmaM * kWmmaK, kWmmaK);
  wmma::load_matrix_sync(b_fragment_0, current_b + warp_column_0, kBlockN);
  wmma::load_matrix_sync(b_fragment_1, current_b + warp_column_1, kBlockN);
  wmma::mma_sync(accumulator_0, a_fragment, b_fragment_0, accumulator_0);
  wmma::mma_sync(accumulator_1, a_fragment, b_fragment_1, accumulator_1);

  if (has_next) {
#if __CUDA_ARCH__ >= 800
    // 当前MMA发射后再等待异步搬运完成，实现搬运和计算的部分重叠
    if (next_is_async) {
      cp_async_wait_all();
    }
#endif
    // 保证next_stage对所有线程可见，同时确保current_stage已经读取完毕
    __syncthreads();
    current_stage = next_stage;
  }
}

// 计算每个warp负责的两个16x16输出fragment位置
const int output_row = block_row + warp_m * kWmmaM;
// 左边一个
const int output_col_0 = block_col + warp_n_group * 2 * kWmmaN;
// 右边一个
const int output_col_1 = output_col_0 + kWmmaN;
const bool full_output = block_row + kBlockM <= m && block_col + kBlockN <= n && n % 8 == 0;

if (full_output) {
  // 完整tile满足WMMA地址和leading dimension要求，直接从寄存器写到global memory


  // wmma::store_matrix_sync 是 WMMA API 中将 Tensor Core accumulator fragment（寄存器中的计算结果）写回 Global Memory 或 Shared Memory 的同步函数。
  // 它是 warp 级操作，自动处理布局转换，适合快速开发，但生产级高性能代码通常用更底层的 stmatrix 或直接寄存器操作来替代。
  wmma::store_matrix_sync(c + static_cast<size_t>(output_row) * n + output_col_0,accumulator_0, n, wmma::mem_row_major);
  wmma::store_matrix_sync(c + static_cast<size_t>(output_row) * n + output_col_1,accumulator_1, n, wmma::mem_row_major);
} else {
  // 边界tile不能直接整块写回，复用8 KiB流水缓冲作为64x64的shared C tile
  __syncthreads();
  half* shared_c = shared_storage;
  wmma::store_matrix_sync(shared_c + warp_m * kWmmaM * kBlockN + warp_n_group * 2 * kWmmaN, accumulator_0, kBlockN, wmma::mem_row_major);
  wmma::store_matrix_sync(shared_c + warp_m * kWmmaM * kBlockN + warp_n_group * 2 * kWmmaN + kWmmaN, accumulator_1, kBlockN, wmma::mem_row_major);
  __syncthreads();
  // 所有线程协作进行带边界判断的shared memory -> global memory写回
  for (int index = tid; index < kBlockM * kBlockN; index += kThreadsPerBlock) {
    const int local_m = index / kBlockN;
    const int local_n = index % kBlockN;
    const int global_m = block_row + local_m;
    const int global_n = block_col + local_n;
    if (global_m < m && global_n < n) {
      c[static_cast<size_t>(global_m) * n + global_n] = shared_c[index];
    }
  }
}
#endif
}

}  // namespace

extern "C" cudaError_t fp16_gemm_tensor_core(const half* a,const half* b,half* c,int m,int n,int k,cudaStream_t stream) {
// 这个核保留了补0和边界写回路径，因此支持任意正整数M/N/K
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
    k <= 0) {
  return cudaErrorInvalidValue;
}

const dim3 block(kThreadsPerBlock);
const dim3 grid((n + kBlockN - 1) / kBlockN, (m + kBlockM - 1) / kBlockM);
// 每个CTA使用256个线程，覆盖一个64x64输出tile
fp16_gemm_tensor_core_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
return cudaPeekAtLastError();
}
