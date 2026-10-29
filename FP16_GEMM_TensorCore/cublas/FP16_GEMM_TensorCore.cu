#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

cudaError_t cuda_error_from_cublas(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

cublasHandle_t cublas_handle(cublasStatus_t* status) {
  static cublasHandle_t handle = nullptr;
  static cublasStatus_t creation_status = [] {
    cublasStatus_t result = cublasCreate(&handle);
    if (result == CUBLAS_STATUS_SUCCESS) {
      result = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    }
    return result;
  }();
  *status = creation_status;
  return handle;
}

}  // namespace

// Row-major C[M,N] = A[M,K] * B[K,N]. cuBLAS is column-major, so compute
// C^T[N,M] = B^T[N,K] * A^T[K,M] without physically transposing the data.
extern "C" cudaError_t fp16_gemm_tensor_core(
    const half* a, const half* b, half* c, int m, int n, int k,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr ||
      m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }

  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  cublasHandle_t handle = cublas_handle(&status);
  if (status != CUBLAS_STATUS_SUCCESS) return cuda_error_from_cublas(status);

  status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return cuda_error_from_cublas(status);

  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  status = cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      b, CUDA_R_16F, n,
      a, CUDA_R_16F, k,
      &beta,
      c, CUDA_R_16F, n,
      CUBLAS_COMPUTE_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return cuda_error_from_cublas(status);
}
