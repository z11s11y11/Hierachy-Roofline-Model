#define GEMM_DEFAULT_M 9216
#define GEMM_DEFAULT_N 10752
#define GEMM_DEFAULT_K 8192
#define GEMM_MIN_SM 70
#define GEMM_DESCRIPTION "64x64 double-buffered WMMA (cp.async on SM80+)"
#include "../FP16_GEMM_TensorCore_test.cuh"
