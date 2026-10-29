#!/usr/bin/env bash
set -euo pipefail

profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$profile_dir/../.." && pwd)
nvcc=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}
ncu=${NCU:-/usr/local/cuda-12.8/bin/ncu}
arch=${CUDA_ARCH:-sm_86}

"$nvcc" -O3 -std=c++17 -arch="$arch" -Xptxas=-v \
  "$root_dir/cublas/FP16_GEMM_TensorCore.cu" \
  "$profile_dir/FP16_GEMM_TensorCore_profile_once.cu" \
  -lcublas -o "$profile_dir/FP16_GEMM_TensorCore_profile_once" \
  2>"$profile_dir/compile.log"

"$ncu" --config-file off --profile-from-start off \
  --kernel-name 'regex:.*gemm.*' --launch-count 1 \
  --section PmSampling --pm-sampling-interval 10000 \
  --replay-mode kernel --cache-control all --clock-control none \
  --export "$profile_dir/cublas_pm_timeline" --force-overwrite \
  "$profile_dir/FP16_GEMM_TensorCore_profile_once"

python3 "$root_dir/mlsys_profile/plot_pipeline_timelines.py" \
  cublas "$profile_dir/cublas_pm_timeline.ncu-rep" "$profile_dir"
