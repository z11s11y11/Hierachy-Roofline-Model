#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kBlockM = 2 * kWmmaM;
constexpr int kBlockN = 2 * kWmmaN;

// A and B are row-major.  Each block computes a 32 x 32 tile of C with four
// warps.  B is transposed only while it is staged in shared memory so that the
// WMMA operation can use the canonical row-major A / column-major B layout.
__global__ void fp16_gemm_tensor_core_kernel(const half* __restrict__ a,const half* __restrict__ b,half* __restrict__ c,int m,int n,int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700

// 16 字节对齐,左矩阵开辟32*16=512个half的共享内存，右矩阵开辟16*32=512个half的共享内存
__shared__ __align__(16) half shared_a[kBlockM * kWmmaK];
__shared__ __align__(16) half shared_b[kBlockN * kWmmaK];

// 一个warp计算一个16x16的tile，4个warp计算一个32x32的tile,一个cta里放4个warp
// 给最后的结果矩阵c也开了4*16*16=1024个half的共享内存
__shared__ __align__(16) half shared_c[kWarpsPerBlock * kWmmaM * kWmmaN];

const int tid = threadIdx.x;
const int warp_id = tid / 32;

//(0,0) (0,1)
//(1,0) (1,1)
const int warp_m = warp_id / 2;
const int warp_n = warp_id % 2;

//每个cta处理的块的起始偏移
const int block_row = blockIdx.y * kBlockM;
const int block_col = blockIdx.x * kBlockN;

// 16*16=256register  a,b,c *3 一个warp的寄存器占用量为768个寄存器
// 声明wmma的寄存器
wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,wmma::row_major> a_fragment;
wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_fragment;
// The accumulator type is deliberately half (not float).
wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> accumulator;
wmma::fill_fragment(accumulator, __float2half(0.0f));

//最外层的循环是k维度的循环，每次处理16个k维度 
for (int k_base = 0; k_base < k; k_base += kWmmaK) {
  // 左矩阵32*16搬运，从gobal memory搬运到shared memory
  for (int index = tid; index < kBlockM * kWmmaK; index += kThreadsPerBlock) {
    const int local_m = index / kWmmaK;
    const int local_k = index % kWmmaK;
    const int global_m = block_row + local_m;
    const int global_k = k_base + local_k;
    shared_a[index] =(global_m < m && global_k < k)? a[static_cast<size_t>(global_m) * k + global_k] : __float2half(0.0f);
  }

  // 右矩阵16*32搬运，从gobal memory搬运到shared memory
  for (int index = tid; index < kBlockN * kWmmaK; index += kThreadsPerBlock) {
    const int local_n = index / kWmmaK;
    const int local_k = index % kWmmaK;
    const int global_n = block_col + local_n;
    const int global_k = k_base + local_k;
    shared_b[index] = (global_n < n && global_k < k) ? b[static_cast<size_t>(global_k) * n + global_n] : __float2half(0.0f);
  }

  // 搬运结束后需要做线程的同步
  __syncthreads();

  // 把数据从shared memory搬运到寄存器中，一个warp搬运对应的数据
  wmma::load_matrix_sync(a_fragment, shared_a + warp_m * kWmmaM * kWmmaK, kWmmaK);
  wmma::load_matrix_sync(b_fragment, shared_b + warp_n * kWmmaN * kWmmaK, kWmmaK);

  // 进行计算和累加
  wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);

  // All warps must finish consuming the tiles before they are overwritten.
  __syncthreads();
}

// k维度走完了，需要把结果从 reg搬运到shared memory
half* warp_result = shared_c + warp_id * kWmmaM * kWmmaN;
wmma::store_matrix_sync(warp_result, accumulator, kWmmaN, wmma::mem_row_major);
__syncthreads();

// A guarded shared-to-global copy supports dimensions not divisible by 16.
for (int index = tid; index < kWarpsPerBlock * kWmmaM * kWmmaN; index += kThreadsPerBlock) {
  const int tile = index / (kWmmaM * kWmmaN);
  const int element = index % (kWmmaM * kWmmaN);
  const int local_m = (tile / 2) * kWmmaM + element / kWmmaN;
  const int local_n = (tile % 2) * kWmmaN + element % kWmmaN;
  const int global_m = block_row + local_m;
  const int global_n = block_col + local_n;
  if (global_m < m && global_n < n) {
    c[static_cast<size_t>(global_m) * n + global_n] = shared_c[index];
  }
}
#endif
}

}  // namespace

// Computes C[m, n] = A[m, k] * B[k, n].  All matrices are row-major and both
// the output and Tensor Core accumulator are FP16.  A stream of nullptr means
// the CUDA default stream.
extern "C" cudaError_t fp16_gemm_tensor_core(const half* a,const half* b,half* c,int m,int n,int k,cudaStream_t stream) {
if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
    k <= 0) {
  return cudaErrorInvalidValue;
}

const dim3 block(kThreadsPerBlock);
const dim3 grid((n + kBlockN - 1) / kBlockN,(m + kBlockM - 1) / kBlockM);
fp16_gemm_tensor_core_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
return cudaPeekAtLastError();
}
