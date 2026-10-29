#!/usr/bin/env bash
set -euo pipefail

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nvcc=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}
arch=${CUDA_ARCH:-sm_86}
cutlass_root=${CUTLASS_ROOT:-/home/zhangshuyi/cutlass}

[[ -x "$nvcc" ]] || { echo "nvcc not found: $nvcc" >&2; exit 1; }
[[ -f "$cutlass_root/include/cutlass/cutlass.h" ]] || {
  echo "CUTLASS headers not found under $cutlass_root; set CUTLASS_ROOT" >&2
  exit 1
}

echo "Compiling CUTLASS FP16 Tensor Core GEMM for $arch"
"$nvcc" -O3 -std=c++17 --expt-relaxed-constexpr -arch="$arch" \
  -I"$cutlass_root/include" \
  "$dir/FP16_GEMM_TensorCore.cu" \
  "$dir/FP16_GEMM_TensorCore_test.cu" \
  -o "$dir/FP16_GEMM_TensorCore_test"

exec "$dir/FP16_GEMM_TensorCore_test" "$@"
