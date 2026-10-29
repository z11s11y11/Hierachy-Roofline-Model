# B LDSM/HMMA dependency-chain experiment

## Configuration

GPU: NVIDIA RTX A6000 (SM86), CUDA 12.8, shape
`M=8064, N=10752, K=8192`.

`FP16_GEMM_B_LDSM_BATCH` selects the B-fragment schedule:

- `0`: baseline; load all six B fragments, then issue the 48-HMMA strip.
- `1`: prefetch one B fragment, then consume the previous one with eight HMMA.
- `2`: prefetch two B fragments, then consume the previous two with 16 HMMA.
- `3`: prefetch three B fragments, then consume the previous three with 24 HMMA.

The pipelined variants use two logical B-register buffers. This gives ptxas a
dependency-chained source schedule instead of six independent source-level B
loads.

## Compilation and SASS

All variants compiled without stack or spill:

| B batch | Registers/thread | Spill |
|---:|---:|---:|
| 0 | 226 | 0 B |
| 1 | 206 | 0 B |
| 2 | 218 | 0 B |
| 3 | 216 | 0 B |

SASS inspection confirms that batches 1--3 move the first HMMA earlier and
interleave later B `LDSM` instructions with the beginning of the HMMA strip.
ptxas does not preserve the source grouping literally: it hoists later loads
into dependency-free HMMA issue slots. The register reduction therefore does
not increase occupancy. The 40 KiB shared-memory allocation still limits the
kernel to two CTAs, or eight warps, per SM.

## Performance

Four 100-iteration rounds were run concurrently on four A6000 GPUs. The batch
assignment was rotated each round so every variant ran once on every GPU. All
variants passed the 128-point validation with the same maximum absolute error.

| B batch | Mean time | Relative to baseline | Mean performance |
|---:|---:|---:|---:|
| 0 | 12.749 ms | baseline | 111.44 TFLOPS |
| 1 | 12.893 ms | 1.13% slower | 110.21 TFLOPS |
| 2 | 12.825 ms | 0.59% slower | 110.77 TFLOPS |
| 3 | 12.749 ms | statistically tied | 111.45 TFLOPS |

An additional same-GPU forward/reverse-order test showed the same ranking:
1B was consistently slower, 2B was close to baseline, and 3B did not produce a
repeatable improvement.

Selected NCU measurements explain the 1B result:

| Metric | Baseline | 1B |
|---|---:|---:|
| Tensor pipe active | 89.73% | 88.57% |
| Short-scoreboard stall | 7.95% | 7.42% |
| MIO-throttle stall | 1.17% | 0.06% |
| Barrier stall | 3.67% | 3.50% |

The 1B regression is not caused by a new LDSM-to-HMMA scoreboard bottleneck.
Instead, interleaving shared-memory instructions breaks up the original
continuous 48-HMMA strip and lowers Tensor-pipeline fill. The lower register
count has no occupancy benefit at the current shared-memory footprint.

## Decision

Keep `FP16_GEMM_B_LDSM_BATCH=0` as the default. The 1B/2B/3B implementations
remain available for experiments with a different CTA geometry or a smaller
shared-memory footprint, where their lower register counts may increase
occupancy.

Example compilation:

```bash
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_86 \
  -DFP16_GEMM_B_LDSM_BATCH=2 \
  FP16_GEMM_TensorCore.cu FP16_GEMM_TensorCore_test.cu \
  -o FP16_GEMM_TensorCore_test
```
