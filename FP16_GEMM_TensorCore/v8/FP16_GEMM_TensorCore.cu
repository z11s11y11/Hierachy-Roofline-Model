#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 连续CTA先沿M方向处理6个tile，再移动N，提高B panel的L2复用率
// GROUP_M=6可以提高cache命中，同时避免更大分组造成L2请求热点
constexpr int kWideGroupedCtaM = 6;

template <int kGroupM>
__device__ __forceinline__ void grouped_cta_tile_coordinates(
    int* tile_m,
    int* tile_n) {
  const int tiles_m = static_cast<int>(gridDim.y);
  const int tiles_n = static_cast<int>(gridDim.x);
  const int physical_tile =
      static_cast<int>(blockIdx.y) * tiles_n + static_cast<int>(blockIdx.x);

  if constexpr (kGroupM == 1) {
    *tile_m = static_cast<int>(blockIdx.y);
    *tile_n = static_cast<int>(blockIdx.x);
  } else {
    const int tiles_per_group = kGroupM * tiles_n;
    const int group_id = physical_tile / tiles_per_group;
    const int first_tile_m = group_id * kGroupM;
    const int actual_group_m =
        min(kGroupM, tiles_m - first_tile_m);
    const int tile_in_group =
        physical_tile - group_id * tiles_per_group;

    // 组内M变化最快，使相邻CTA访问同一B panel；一组完成后N再向前移动
    *tile_m = first_tile_m + tile_in_group % actual_group_m;
    *tile_n = tile_in_group / actual_group_m;
  }
}

// 192x192 CTA tile、2x2 warp网格、4级K=16异步流水线
constexpr int kFastBlockM = 192;
// K panel按16x16 tile做XOR swizzle，每个tile紧凑存放32个16字节向量
constexpr int kFastAStride = kWmmaK;
constexpr int kFastAStageElements = kFastBlockM * kFastAStride;
constexpr int kFastAStages = 4;
constexpr int kFastBStages = 4;

// N方向固定为192；2x2 warp网格中每个warp计算96x96输出
constexpr int kWideBlockN = 192;
static_assert(kWideBlockN >= 32 && kWideBlockN % (2 * kWmmaN) == 0,
              "wide N must be a positive multiple of 32");
constexpr int kWideTilesPerWarpM = 6;
constexpr int kWideWarpRows =
    kFastBlockM / (kWideTilesPerWarpM * kWmmaM);
constexpr int kWideWarpCols = 2;
static_assert(kWideWarpCols >= 1,
              "wide warp columns must be positive");
static_assert(kWideBlockN % (kWideWarpCols * kWmmaN) == 0,
              "wide N must be divisible by warp columns times 16");
constexpr int kWideWarps = kWideWarpRows * kWideWarpCols;
constexpr int kWideThreads = kWideWarps * 32;
constexpr int kWideTilesPerWarpN = kWideBlockN / (kWideWarpCols * kWmmaN);
constexpr int kWideBStride = kWideBlockN;
constexpr int kWideBStageElements = kWmmaK * kWideBStride;
constexpr int kWideSharedElements =
    kFastAStages * kFastAStageElements +
    kFastBStages * kWideBStageElements;
constexpr int kWideSharedBytes = kWideSharedElements * sizeof(half);
constexpr int kWideBVectorsPerRow = kWideBlockN / 8;
constexpr int kWideBCopyVectors = kWmmaK * kWideBVectorsPerRow;
static_assert(kWideSharedBytes == 48 * 1024,
              "192x192x16 four-stage tile must use 48 KiB");

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

__device__ __forceinline__ unsigned int shared_address(
    const void* pointer) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

// physical = logical ^ (row >> 2)，把ldmatrix访问的8行分散到不同bank group
__device__ __forceinline__ int swizzled_tile_vector(int row, int vector) {
  const int logical_vector = row * 2 + vector;
  return logical_vector ^ (row >> 2);
}

__device__ __forceinline__ half* swizzled_tile_pointer(
    half* stage, int tile, int row, int vector) {
  constexpr int kTileElements = kWmmaM * kWmmaK;
  constexpr int kVectorElements = 8;
  return stage + tile * kTileElements +
         swizzled_tile_vector(row, vector) * kVectorElements;
}

__device__ __forceinline__ half* swizzled_a_pointer(
    half* stage, int row, int column) {
  return swizzled_tile_pointer(stage, row / kWmmaM, row % kWmmaM,
                               column / 8);
}

__device__ __forceinline__ half* swizzled_b_pointer(
    half* stage, int row, int column) {
  return swizzled_tile_pointer(stage, column / kWmmaN, row,
                               (column % kWmmaN) / 8);
}

__device__ __forceinline__ void ldmatrix_a_x4(
    const half* pointer, unsigned int (&fragment)[4]) {
  const unsigned int address = shared_address(pointer);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
        "=r"(fragment[3])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void ldmatrix_b_x4_trans(
    const half* pointer, unsigned int (&fragment)[4]) {
  const unsigned int address = shared_address(pointer);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
        "=r"(fragment[3])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    unsigned int (&accumulator)[2],
    const unsigned int (&a_fragment)[4],
    const unsigned int& b_fragment_0,
    const unsigned int& b_fragment_1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
      : "+r"(accumulator[0]), "+r"(accumulator[1])
      : "r"(a_fragment[0]), "r"(a_fragment[1]), "r"(a_fragment[2]),
        "r"(a_fragment[3]), "r"(b_fragment_0), "r"(b_fragment_1));
}

#endif

__global__ __launch_bounds__(kWideThreads, 2) void fp16_gemm_outer_product_sm86_wide_pipeline_kernel(const half* __restrict__ a,const half* __restrict__ b,half* __restrict__ c,int n,int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// swizzled A/B四级缓冲合计48 KiB，使用动态shared memory
extern __shared__ __align__(32) half shared_storage[];
half* const shared_a = shared_storage;
half* const shared_b =
    shared_storage + kFastAStages * kFastAStageElements;

const int tid = threadIdx.x;
const int warp_id = tid / 32;

// 4个warp按2x2排列，每个warp负责6x6个16x16 C fragment（96x96输出）
const int warp_m = warp_id / kWideWarpCols;
const int warp_n = warp_id % kWideWarpCols;

// GROUP_M=6把物理blockIdx重排为L2更友好的逻辑tile坐标
int tile_m = 0;
int tile_n = 0;
grouped_cta_tile_coordinates<kWideGroupedCtaM>(&tile_m, &tile_n);
const int block_row = tile_m * kFastBlockM;
const int block_col = tile_n * kWideBlockN;

// 保存ldmatrix/mma.sync的原始32位寄存器；B fragment完整驻留，不使用较慢的streaming方案
unsigned int a_fragments[kWideTilesPerWarpM][4];
unsigned int b_fragments[kWideTilesPerWarpN][4];
unsigned int accumulators[kWideTilesPerWarpM][kWideTilesPerWarpN * 2][2] = {};

// A每个stage有384个16字节向量，128个线程各搬3个；B向量也在全部线程间轮转分配
constexpr int kWideACopyVectors = kFastBlockM * kWmmaK / 8;

#pragma unroll
// 预取前3个K tile，为四级cp.async流水线预热
for (int stage = 0; stage < 3; ++stage) {
  half* const stage_a = shared_a + stage * kFastAStageElements;
  half* const stage_b = shared_b + stage * kWideBStageElements;
  for (int copy = tid; copy < kWideACopyVectors;
       copy += kWideThreads) {
    const int a_row = copy / 2;
    const int a_column = (copy % 2) * 8;
    cp_async_16(swizzled_a_pointer(stage_a, a_row, a_column),
                a + static_cast<size_t>(block_row + a_row) * k +
                    stage * kWmmaK + a_column);
  }
#pragma unroll
  for (int copy = tid; copy < kWideBCopyVectors;
       copy += kWideThreads) {
    const int b_row = copy / kWideBVectorsPerRow;
    const int b_column = (copy % kWideBVectorsPerRow) * 8;
    cp_async_16(swizzled_b_pointer(stage_b, b_row, b_column),
                b + static_cast<size_t>(stage * kWmmaK + b_row) * n +
                    block_col + b_column);
  }
  cp_async_commit();
}
cp_async_wait_group_2();
__syncthreads();

const int k_steps = k / kWmmaK;
// 消费当前stage，同时预取step+3，实现global memory搬运和Tensor Core计算重叠
for (int step = 0; step < k_steps; ++step) {
  half* const current_a =
      shared_a + (step % kFastAStages) * kFastAStageElements;
  half* const current_b =
      shared_b + (step % kFastBStages) * kWideBStageElements;
  const int lane_id = tid % 32;
#pragma unroll
  // ldmatrix.x4读取经过swizzle的A tile
  for (int tile_m = 0; tile_m < kWideTilesPerWarpM; ++tile_m) {
    const int a_tile = warp_m * kWideTilesPerWarpM + tile_m;
    ldmatrix_a_x4(
        swizzled_tile_pointer(current_a, a_tile, lane_id % 16,
                              lane_id / 16),
        a_fragments[tile_m]);
  }
#pragma unroll
  // ldmatrix.x4.trans读取并转置B tile，B fragment全部常驻寄存器
  for (int tile_n = 0; tile_n < kWideTilesPerWarpN; ++tile_n) {
    const int b_tile = warp_n * kWideTilesPerWarpN + tile_n;
    ldmatrix_b_x4_trans(
        swizzled_tile_pointer(current_b, b_tile, lane_id % 16,
                              lane_id / 16),
        b_fragments[tile_n]);
  }

  // 当前stage已进入寄存器，可以复用它的shared memory槽位加载未来tile
  const int future_step = step + 3;
  half* const future_a =
      shared_a + (future_step % kFastAStages) * kFastAStageElements;
  half* const future_b =
      shared_b + (future_step % kFastBStages) * kWideBStageElements;
  if (future_step < k_steps) {
#pragma unroll
    for (int copy = tid; copy < kWideACopyVectors;
         copy += kWideThreads) {
      const int a_row = copy / 2;
      const int a_column = (copy % 2) * 8;
      cp_async_16(
          swizzled_a_pointer(future_a, a_row, a_column),
          a + static_cast<size_t>(block_row + a_row) * k +
              future_step * kWmmaK + a_column);
    }
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors;
         copy += kWideThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_16(
          swizzled_b_pointer(future_b, b_row, b_column),
          b + static_cast<size_t>(future_step * kWmmaK + b_row) * n +
              block_col + b_column);
    }
  } else {
#pragma unroll
    for (int copy = tid; copy < kWideACopyVectors;
         copy += kWideThreads) {
      const int a_row = copy / 2;
      const int a_column = (copy % 2) * 8;
      cp_async_zero_16(swizzled_a_pointer(future_a, a_row, a_column));
    }
#pragma unroll
    for (int copy = tid; copy < kWideBCopyVectors;
         copy += kWideThreads) {
      const int b_row = copy / kWideBVectorsPerRow;
      const int b_column = (copy % kWideBVectorsPerRow) * 8;
      cp_async_zero_16(swizzled_b_pointer(future_b, b_row, b_column));
    }
  }
  cp_async_commit();

  // m16n8k16每次计算半个16x16 tile，累加器全程保留在寄存器
#pragma unroll
  for (int tile_m = 0; tile_m < kWideTilesPerWarpM; ++tile_m) {
#pragma unroll
    for (int tile_n8 = 0; tile_n8 < kWideTilesPerWarpN * 2; ++tile_n8) {
      const int tile_n = tile_n8 / 2;
      const int n_half = tile_n8 % 2;
      mma_m16n8k16_f16(accumulators[tile_m][tile_n8],
                       a_fragments[tile_m],
                       b_fragments[tile_n][n_half * 2],
                       b_fragments[tile_n][n_half * 2 + 1]);
    }
  }

  cp_async_wait_group_2();
  __syncthreads();
}

const int lane_row = (tid % 32) / 4;
const int lane_column = ((tid % 32) % 4) * 2;
#pragma unroll
// 每个线程用一次32位store写回两个FP16结果
for (int tile_m = 0; tile_m < kWideTilesPerWarpM; ++tile_m) {
#pragma unroll
  for (int tile_n8 = 0; tile_n8 < kWideTilesPerWarpN * 2; ++tile_n8) {
    const int output_row =
        block_row + (warp_m * kWideTilesPerWarpM + tile_m) * kWmmaM;
    const int output_column =
        block_col + warp_n * kWideTilesPerWarpN * 2 * 8 +
        tile_n8 * 8 + lane_column;
    *reinterpret_cast<unsigned int*>(
        c + static_cast<size_t>(output_row + lane_row) * n +
            output_column) = accumulators[tile_m][tile_n8][0];
    *reinterpret_cast<unsigned int*>(
        c + static_cast<size_t>(output_row + lane_row + 8) * n +
            output_column) = accumulators[tile_m][tile_n8][1];
  }
}
#endif
}

cudaError_t configure_sm86_wide_pipeline_kernel() {
  // 第一次调用时设置48 KiB动态shared memory和shared memory优先的carveout
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
// 最终核只支持完整tile，M/N/K必须分别是192/192/64的倍数
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
    k <= 0 || m % kFastBlockM != 0 || n % kWideBlockN != 0 ||
    k % (kFastAStages * kWmmaK) != 0) {
  return cudaErrorInvalidValue;
}

const cudaError_t error = configure_sm86_wide_pipeline_kernel();
if (error != cudaSuccess) return error;
const dim3 grid(n / kWideBlockN, m / kFastBlockM);
fp16_gemm_outer_product_sm86_wide_pipeline_kernel
    <<<grid, kWideThreads, kWideSharedBytes, stream>>>(a, b, c, n, k);
return cudaPeekAtLastError();
}
