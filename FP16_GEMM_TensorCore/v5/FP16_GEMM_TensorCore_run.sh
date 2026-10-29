#!/usr/bin/env bash

export CUDA_HOME=/usr/local/cuda-12.8
export NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"


set -euo pipefail
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$dir/../run_gemm.sh" "$dir" "$@"
