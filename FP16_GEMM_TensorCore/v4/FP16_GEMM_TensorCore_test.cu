#define GEMM_DEFAULT_M 8064
#define GEMM_DEFAULT_N 10752
#define GEMM_DEFAULT_K 8192
#define GEMM_MIN_SM 80
#define GEMM_DESCRIPTION "128x192 four-stage cp.async pipeline"
#include "../FP16_GEMM_TensorCore_test.cuh"
