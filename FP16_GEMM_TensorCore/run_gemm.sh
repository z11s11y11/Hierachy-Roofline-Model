#!/usr/bin/env bash
set -euo pipefail

dir=${1:?missing source directory}
shift
nvcc=${NVCC:-$(command -v nvcc || true)}
if [[ -z "$nvcc" && -x /usr/local/cuda/bin/nvcc ]]; then
  nvcc=/usr/local/cuda/bin/nvcc
fi
[[ -n "$nvcc" ]] || { echo "nvcc not found; set NVCC" >&2; exit 1; }

arch=${CUDA_ARCH:-sm_80}
if [[ -z ${CUDA_ARCH:-} ]] && command -v nvidia-smi >/dev/null; then
  capability=$(nvidia-smi --id=0 --query-gpu=compute_cap \
    --format=csv,noheader 2>/dev/null | tr -d '.[:space:]' || true)
  [[ $capability =~ ^[0-9]+$ ]] && arch=sm_$capability
fi

echo "Compiling $(basename "$dir") for $arch"
"$nvcc" -O3 -std=c++17 -arch="$arch" -Xptxas=-v \
  "$dir/FP16_GEMM_TensorCore.cu" "$dir/FP16_GEMM_TensorCore_test.cu" \
  -o "$dir/FP16_GEMM_TensorCore_test"
exec "$dir/FP16_GEMM_TensorCore_test" "$@"
