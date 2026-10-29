#!/usr/bin/env bash
set -euo pipefail

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nvcc=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}
arch=${CUDA_ARCH:-sm_86}

echo "Compiling cuBLAS FP16 Tensor Core GEMM for $arch"
"$nvcc" -O3 -std=c++17 -arch="$arch" \
  "$dir/FP16_GEMM_TensorCore.cu" \
  "$dir/FP16_GEMM_TensorCore_test.cu" \
  -lcublas -o "$dir/FP16_GEMM_TensorCore_test"

exec "$dir/FP16_GEMM_TensorCore_test" "$@"
