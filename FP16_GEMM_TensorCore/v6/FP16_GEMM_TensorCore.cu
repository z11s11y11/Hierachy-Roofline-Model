// ldmatrix 的实际访问模式是按列协作加载子矩阵，Swizzle 的设计目标是让同一列的连续行元素分散到不同 bank，而不是让不同 row 的相同 vec 不冲突。
// XOR Swizzle 的 (row >> 2) 正是为了让 column-wise 访问时，每 4 行切换一次 bank 分布模式，消除 column 访问的 bank conflict。

// Column 0 的元素 (row 0-15):
// Row 0:  swizzled = 0^0 = 0  → bank 0,1,2,3
// Row 1:  swizzled = 2^0 = 2  → bank 8,9,10,11
// Row 2:  swizzled = 4^0 = 4  → bank 16,17,18,19
// Row 3:  swizzled = 6^0 = 6  → bank 24,25,26,27
// Row 4:  swizzled = 8^1 = 9  → bank 4,5,6,7   (9*4=36%32=4)
// Row 5:  swizzled = 10^1 = 11 → bank 12,13,14,15 (11*4=44%32=12)
// Row 6:  swizzled = 12^1 = 13 → bank 20,21,22,23
// Row 7:  swizzled = 14^1 = 15 → bank 28,29,30,31

// Column 0 的 8 行分散到: 0, 8, 16, 24, 4, 12, 20, 28 (bank 起始)
// → 无冲突！✓
// 所以说涉及到两步：从全局内存中取数搬运到共享内存的时候就分开放
// reg从共享内存中取数的时候也要按照对应的法则来取数

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 连续CTA先沿M方向处理6个tile，再移动到下一个N tile，提高B panel的L2复用率
// GROUP_M=6是紧凑swizzle版本在RTX A6000上的实测最优配置
constexpr int kWideGroupedCtaM = 6;

template <int kGroupM>
__device__ __forceinline__ void grouped_cta_tile_coordinates( int* tile_m, int* tile_n) {
  // M N 方向总 tile 数
  const int tiles_m = static_cast<int>(gridDim.y);
  const int tiles_n = static_cast<int>(gridDim.x);
  // 当前 CTA 的物理序号
  // 从左到右，再从上往下数，是第几个cta
  const int physical_tile = static_cast<int>(blockIdx.y) * tiles_n + static_cast<int>(blockIdx.x);

  if constexpr (kGroupM == 1) {
    // 直接使用原始坐标
    *tile_m = static_cast<int>(blockIdx.y);
    *tile_n = static_cast<int>(blockIdx.x);
  } else {
    // tiles_n是一整行有多少个tile，kGroupM代表几行打包成一组
    const int tiles_per_group = kGroupM * tiles_n;  // 每组包含的 tile 数
    const int group_id = physical_tile / tiles_per_group;  // 当前属于第几组
    const int first_tile_m = group_id * kGroupM;  // 组起始 M 位置
    const int actual_group_m = min(kGroupM, tiles_m - first_tile_m);  // 边界处理
    // 在组里面属于第几个tile
    const int tile_in_group = physical_tile - group_id * tiles_per_group;  // 组内偏移

    // 组内M变化最快，使相邻CTA访问相同的B panel；一组M完成后N再向前移动
    *tile_m = first_tile_m + tile_in_group % actual_group_m;
    *tile_n = tile_in_group / actual_group_m;

    // 先沿m维度运动，再沿n维度运动
  }
}

// 一个CTA处理128x192输出tile，使用8个warp和4级K=16异步流水线
constexpr int kFastBlockM = 128;

// 一个cta里面放置8个warp
constexpr int kFastWarps = 8;
constexpr int kFastThreads = kFastWarps * 32;

constexpr int kFastAStride = kWmmaK;

// 左矩阵放在shared memory中是128*16half=4kB
constexpr int kFastAStageElements = kFastBlockM * kFastAStride;
constexpr int kFastAStages = 4;
constexpr int kFastBStages = 4;
constexpr int kWideBlockN = 192;
constexpr int kWideTilesPerWarpN = 6;
constexpr int kWideBStride = kWideBlockN;
// 右矩阵放在shared memory中是16*192half=6kB
constexpr int kWideBStageElements = kWmmaK * kWideBStride;
constexpr int kWideSharedElements = kFastAStages * kFastAStageElements + kFastBStages * kWideBStageElements;
constexpr int kWideSharedBytes = kWideSharedElements * sizeof(half);
constexpr int kWideBVectorsPerRow = kWideBlockN / 8;
constexpr int kWideBCopyVectors = kWmmaK * kWideBVectorsPerRow;
static_assert(kWideSharedElements * sizeof(half) == 40 * 1024);
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

__device__ __forceinline__ unsigned int shared_address( const void* pointer) {

  // 将 Generic 地址空间的指针转换为 Shared Memory 地址空间的偏移，供 PTX 指令使用。
  //   PTX 指令的操作数有明确的地址空间约束：
  // ldmatrix 的 [%4] 必须是 shared 空间地址
  // cp.async 的 [%0]（目标）必须是 shared，[%1]（源）必须是 global
  return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

// 一个16x16 half tile包含32个16字节向量
// physical = logical ^ (row >> 2)，让ldmatrix访问的8行落到不同bank group
__device__ __forceinline__ int swizzled_tile_vector(int row, int vector) {
  const int logical_vector = row * 2 + vector;
  // 目前理解下来是每4轮左右翻转一下
  return logical_vector ^ (row >> 2);
}
  // 全局内存--> shared memory
  // 0	1		  0	  1
  // 2	3		  2	  3
  // 4	5		  4	  5
  // 6	7		  6	  7
  // 8	9		  9	  8
  // 10	11		11	10
  // 12	13		13	12
  // 14	15		15	14
  // 16	17		18	19
  // 18	19		16	17
  // 20	21		22	23
  // 22	23		20	21
  // 24	25		27	26
  // 26	27		25	24
  // 28	29		31	30
  // 30	31		29	28




//row表示是16*16的这个tile中的第几行，vector只有0，1可选，因为一个线程一次访问了8个pf16=16byte的数据
// tile表示的是左矩阵a 128*16划分成16*16的tile后的第几个tile
// stage表示的是左矩阵a 128*16在shared memory中的起始地址
__device__ __forceinline__ half* swizzled_tile_pointer( half* stage, int tile, int row, int vector) {
  // kTileElements代表16*16的tile一共有多少个元素
  constexpr int kTileElements = kWmmaM * kWmmaK;

  // kVectorElements代表一个线程取8个pf16=16byte的数据
  constexpr int kVectorElements = 8;

  // stage + tile * kTileElements代表每个tile的起始地址
  return stage + tile * kTileElements + swizzled_tile_vector(row, vector) * kVectorElements;
}

__device__ __forceinline__ half* swizzled_a_pointer( half* stage, int row, int column) {
  // 调用swizzled_tile_pointer
  // stage = stage_a,每一个stage对应左矩阵的起始地址
  // row = a_row, column = a_column。
  // 8个war对应8*32=128*2个16byte取数
  // (0,0),(0,8)
  // (1,0),(1,8)
  // (2,0),(2,8)
  // ...
  // (128,0),(128,8)
  // row / kWmmaM对应第几组，128行一共可以分8组
  // row % kWmmaM代表组内的第几行
  return swizzled_tile_pointer(stage, row / kWmmaM, row % kWmmaM, column / 8);
}

__device__ __forceinline__ half* swizzled_b_pointer(half* stage, int row, int column) {
  return swizzled_tile_pointer(stage, column / kWmmaN, row, (column % kWmmaN) / 8);
}


// ldmatrix_a_x4应该是把数据从shared memory放入寄存器中
__device__ __forceinline__ void ldmatrix_a_x4( const half* pointer, unsigned int (&fragment)[4]) {
  const unsigned int address = shared_address(pointer);

  // ldmatrix 加载操作 ldmatrix = load matrix
  // sync warp 同步加载
  // aligned 16 字节对齐
  // m8n8 8×8 矩阵形状
  // x4 加载 4 个 8×8 矩阵
  // shared 存储空间: shared = shared memory
  // b16 数据类型: b16 = 16-bit = half
  asm volatile( "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
    : "r"(address)
    : "memory");
    // "=r"	输出约束，= 表示只写，r = 32-bit 通用寄存器
    // "r"	输入约束，r = 32-bit 通用寄存器
    // "l"	输入约束，l = 64-bit 通用寄存器（用于地址）
}

__device__ __forceinline__ void ldmatrix_b_x4_trans( const half* pointer, unsigned int (&fragment)[4]) {
  const unsigned int address = shared_address(pointer);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void mma_m16n8k16_f16( unsigned int (&accumulator)[2], const unsigned int (&a_fragment)[4], const unsigned int& b_fragment_0,
    const unsigned int& b_fragment_1) {
      // mma  MMA 操作
      // m16n8k16 m维度= 16, n维度= 8, k维度= 16
      // +r 读写
      // r  只读
      // A 矩阵布局: row = row-major
      // B 矩阵布局: col = column-major
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
      : "+r"(accumulator[0]), "+r"(accumulator[1])
      : "r"(a_fragment[0]), "r"(a_fragment[1]), "r"(a_fragment[2]), "r"(a_fragment[3]), 
      "r"(b_fragment_0), "r"(b_fragment_1));
}
#endif


__global__ __launch_bounds__(kFastThreads, 2) void fp16_gemm_outer_product_sm86_wide_pipeline_kernel(const half* __restrict__ a,const half* __restrict__ b,half* __restrict__ c,int n,int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// 4级swizzled A/B缓冲总共40 KiB，使用动态shared memory统一管理
extern __shared__ __align__(32) half shared_storage[];


// shared_a表示在一个cta中，左矩阵缓存的起始地址
half* const shared_a = shared_storage;

// shared_b表示在一个cta中，右矩阵缓存的起始地址
half* const shared_b = shared_storage + kFastAStages * kFastAStageElements;

const int tid = threadIdx.x;
const int warp_id = tid / 32;

// 8个warp按4x2排列，每个warp计算2x6个16x16 C fragment
const int warp_m = warp_id / 2;
const int warp_n = warp_id % 2;

// GROUP_M=6重新映射物理CTA坐标，使相邻CTA复用同一块B数据
int tile_m = 0;
int tile_n = 0;
grouped_cta_tile_coordinates<kWideGroupedCtaM>(&tile_m, &tile_n);
const int block_row = tile_m * kFastBlockM;
const int block_col = tile_n * kWideBlockN;

// 直接保存ldmatrix和mma.sync使用的32位寄存器，不再经过WMMA抽象

// 下面参数的含义解释：
//                 196 
//        warp 0        warp 1
// 128.   warp 2        warp 3
//        warp 4        warp 5
//        warp 6        warp 7
// 对于一个warp 来说，要处理32*96的数据，32=2*16，96=6（kWideTilesPerWarpN）*16
//  每一个16*16的数据块都被拆分成4个8*8

unsigned int a_fragments[2][4];
unsigned int b_fragments[kWideTilesPerWarpN][4];

// 采用m=16,n=8,k=16的矩阵乘方法，对于一个warp来说，需要2*12（kWideTilesPerWarpN * 2）的 16*8的累加器，但我估计累加器也是8*8的方阵，所以末尾还需要一个2
unsigned int accumulators[2][kWideTilesPerWarpN * 2][2] = {};

// 每个stage的A有128*2=256个16字节向量，B有2*192=384个；前128个线程多搬一个B向量
const int a_row = tid / 2;
const int a_column = (tid % 2) * 8;

#pragma unroll
// 先预取3个K tile，写入经过XOR swizzle计算后的shared memory地址
for (int stage = 0; stage < 3; ++stage) {

  // 每个stage的起始地址
  half* const stage_a = shared_a + stage * kFastAStageElements;
  half* const stage_b = shared_b + stage * kWideBStageElements;

  // a + static_cast<size_t>(block_row + a_row) * k + stage * kWmmaK + a_column 表示全局内存中左矩阵的地址
  // swizzled_a_pointer(stage_a, a_row, a_column) 表示经过XOR swizzle计算后的shared memory地址
  cp_async_16(swizzled_a_pointer(stage_a, a_row, a_column), a + static_cast<size_t>(block_row + a_row) * k + stage * kWmmaK + a_column);
  // 全局内存--> shared memory
  // 0	1		  0	  1
  // 2	3		  2	  3
  // 4	5		  4	  5
  // 6	7		  6	  7
  // 8	9		  9	  8
  // 10	11		11	10
  // 12	13		13	12
  // 14	15		15	14

#pragma unroll
  for (int copy = tid; copy < kWideBCopyVectors; copy += kFastThreads) {
    const int b_row = copy / kWideBVectorsPerRow;
    const int b_column = (copy % kWideBVectorsPerRow) * 8;
    cp_async_16(swizzled_b_pointer(stage_b, b_row, b_column), b + static_cast<size_t>(stage * kWmmaK + b_row) * n + block_col + b_column);
  }
  cp_async_commit();
}
cp_async_wait_group_2();
__syncthreads();

const int k_steps = k / kWmmaK;
// 每轮从当前stage取数，同时把step+3预取到刚释放的环形槽位
for (int step = 0; step < k_steps; ++step) {
  half* const current_a = shared_a + (step % kFastAStages) * kFastAStageElements;
  half* const current_b = shared_b + (step % kFastBStages) * kWideBStageElements;
  const int lane_id = tid % 32;
#pragma unroll
  // ldmatrix.x4把swizzled A tile从shared memory搬到寄存器
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
    const int a_tile = warp_m * 2 + tile_m;
    // a_fragments[2][4]
    // swizzled_tile_pointer(current_a, a_tile, lane_id % 16, lane_id / 16) 代表shared memory中的地址
    // a_fragments[tile_m] 代表寄存器中的地址，而且是分了4块，a_fragments[tile_m][0]~a_fragments[tile_m][3]
    ldmatrix_a_x4( swizzled_tile_pointer(current_a, a_tile, lane_id % 16, lane_id / 16), a_fragments[tile_m]);
  }
#pragma unroll
  // ldmatrix.x4.trans搬运B，并完成mma需要的转置布局
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    const int b_tile = warp_n * kWideTilesPerWarpN + tile_n;
    ldmatrix_b_x4_trans( swizzled_tile_pointer(current_b, b_tile, lane_id % 16, lane_id / 16), b_fragments[tile_n]);
  }

  // 异步加载未来K tile，和当前寄存器上的mma.sync重叠
  const int future_step = step + 3;
  half* const future_a = shared_a + (future_step % kFastAStages) * kFastAStageElements;
  half* const future_b = shared_b + (future_step % kFastBStages) * kWideBStageElements;
  half* const future_a_destination = swizzled_a_pointer(future_a, a_row, a_column);
  if (future_step < k_steps) {
    cp_async_16( future_a_destination, a + static_cast<size_t>(block_row + a_row) * k + future_step * kWmmaK + a_column);
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors; copy += kFastThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_16( swizzled_b_pointer(future_b, b_row, b_column), b + static_cast<size_t>(future_step * kWmmaK + b_row) * n + block_col + b_column);
    }
  } else {
    cp_async_zero_16(future_a_destination);
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors; copy += kFastThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_zero_16(swizzled_b_pointer(future_b, b_row, b_column));
    }
  }
  cp_async_commit();

#pragma unroll
  // m16n8k16一次计算半个16x16 tile，所以N方向循环数量乘2
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
    for (int tile_n8 = 0; tile_n8 < kWideTilesPerWarpN * 2; ++tile_n8) {
      const int tile_n = tile_n8 / 2;
      const int n_half = tile_n8 % 2;
      mma_m16n8k16_f16(accumulators[tile_m][tile_n8], a_fragments[tile_m], b_fragments[tile_n][n_half * 2], b_fragments[tile_n][n_half * 2 + 1]);
    }
  }

  cp_async_wait_group_2();
  __syncthreads();
}

const int lane_row = (tid % 32) / 4;
const int lane_column = ((tid % 32) % 4) * 2;
#pragma unroll
// 每个线程一次写回32 bit（两个half），把寄存器中的C结果合并写到global memory
for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
  for (int tile_n8 = 0; tile_n8 < kWideTilesPerWarpN * 2; ++tile_n8) {
    const int output_row = block_row + (warp_m * 2 + tile_m) * kWmmaM;
    const int output_column = block_col + warp_n * kWideTilesPerWarpN * 2 * 8 + tile_n8 * 8 + lane_column;
    *reinterpret_cast<unsigned int*>( c + static_cast<size_t>(output_row + lane_row) * n + output_column) = accumulators[tile_m][tile_n8][0];
    *reinterpret_cast<unsigned int*>( c + static_cast<size_t>(output_row + lane_row + 8) * n + output_column) = accumulators[tile_m][tile_n8][1];
  }
}
#endif
}

cudaError_t configure_sm86_wide_pipeline_kernel() {
  // 第一次调用时设置动态shared memory大小和shared memory优先的carveout
  static const cudaError_t configuration_error = [] {
    cudaError_t error = cudaFuncSetAttribute( fp16_gemm_outer_product_sm86_wide_pipeline_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kWideSharedBytes);
    if (error != cudaSuccess) return error;
    return cudaFuncSetAttribute( fp16_gemm_outer_product_sm86_wide_pipeline_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  }();
  return configuration_error;
}

}  // namespace

// Computes row-major C[M, N] = A[M, K] * B[K, N].  Inputs, output, and the
// Tensor Core accumulators are FP16.  Passing nullptr selects the default CUDA
// stream.  The selected kernel requires M/N to match its CTA tile and K to be
// a multiple of the four-stage pipeline depth.
extern "C" cudaError_t fp16_gemm_tensor_core(const half* a,const half* b,half* c,int m,int n,int k,cudaStream_t stream) {
// 只保留128x192、GROUP_M=6、XOR swizzle核，尺寸必须按128/192/64对齐
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
    k <= 0 || m % kFastBlockM != 0 || n % kWideBlockN != 0 ||
    k % (kFastAStages * kWmmaK) != 0) {
  return cudaErrorInvalidValue;
}

const cudaError_t error = configure_sm86_wide_pipeline_kernel();
if (error != cudaSuccess) return error;
const dim3 grid(n / kWideBlockN, m / kFastBlockM);
fp16_gemm_outer_product_sm86_wide_pipeline_kernel <<<grid, kFastThreads, kWideSharedBytes, stream>>>(a, b, c, n, k);
return cudaPeekAtLastError();
}
