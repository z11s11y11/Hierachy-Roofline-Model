#define GEMM_DEFAULT_M 8064
#define GEMM_DEFAULT_N 10752
#define GEMM_DEFAULT_K 8192
#define GEMM_MIN_SM 80
#define GEMM_DESCRIPTION \
  "192x192x16 CTA, 96x96 warp, 4-stage XOR-swizzled pipeline"
#include "../FP16_GEMM_TensorCore_test.cuh"
