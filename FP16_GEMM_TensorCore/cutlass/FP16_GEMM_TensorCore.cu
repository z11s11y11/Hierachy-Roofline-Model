#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

namespace {

using Element = cutlass::half_t;
using Layout = cutlass::layout::ColumnMajor;

// The benchmark shape becomes N x M x K after taking the transpose view.
// 256 and 128 exactly divide the default N=10752 and M=8064 respectively.
using CutlassGemm = cutlass::gemm::device::Gemm<
    Element, Layout,
    Element, Layout,
    Element, Layout,
    Element,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        Element, 8, Element, Element>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

cudaError_t cuda_error_from_cutlass(cutlass::Status status) {
  if (status == cutlass::Status::kSuccess) return cudaSuccess;
  if (status == cutlass::Status::kErrorInvalidProblem ||
      status == cutlass::Status::kErrorMisalignedOperand) {
    return cudaErrorInvalidValue;
  }
  return cudaErrorUnknown;
}

}  // namespace

// Row-major C[M,N] = A[M,K] * B[K,N]. Without moving data, view the
// row-major buffers as column-major transposes and compute
// C^T[N,M] = B^T[N,K] * A^T[K,M].
extern "C" cudaError_t fp16_gemm_tensor_core(
    const half* a, const half* b, half* c, int m, int n, int k,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr ||
      m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }

  static_assert(sizeof(half) == sizeof(Element),
                "CUDA and CUTLASS FP16 types must have the same size");
  const auto* cutlass_a = reinterpret_cast<const Element*>(b);
  const auto* cutlass_b = reinterpret_cast<const Element*>(a);
  auto* cutlass_c = reinterpret_cast<Element*>(c);

  const Element alpha = Element(1.0f);
  const Element beta = Element(0.0f);
  typename CutlassGemm::Arguments arguments(
      cutlass::gemm::GemmCoord(n, m, k),
      {cutlass_a, n},
      {cutlass_b, k},
      {cutlass_c, n},
      {cutlass_c, n},
      {alpha, beta});

  const cutlass::Status support = CutlassGemm::can_implement(arguments);
  if (support != cutlass::Status::kSuccess) {
    return cuda_error_from_cutlass(support);
  }

  CutlassGemm gemm;
  return cuda_error_from_cutlass(gemm(arguments, nullptr, stream));
}
