#define GEMM_DEFAULT_M 8064
#define GEMM_DEFAULT_N 10752
#define GEMM_DEFAULT_K 8192
#define GEMM_MIN_SM 80
#define GEMM_DESCRIPTION "CUTLASS FP16 Tensor Core GEMM (256x128x32, FP16 accumulation)"
#include "../FP16_GEMM_TensorCore_test.cuh"
