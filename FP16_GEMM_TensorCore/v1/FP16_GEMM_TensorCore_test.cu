#define GEMM_DEFAULT_M 8192
#define GEMM_DEFAULT_N 8192
#define GEMM_DEFAULT_K 8192
#define GEMM_MIN_SM 70
#define GEMM_DESCRIPTION "32x32 shared-memory WMMA baseline"
#include "../FP16_GEMM_TensorCore_test.cuh"
