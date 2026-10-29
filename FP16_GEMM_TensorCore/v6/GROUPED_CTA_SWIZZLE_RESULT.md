# v20 L2 grouped CTA swizzle result

## Change

- Added a compile-time `FP16_GEMM_GROUP_M` mapping to the 128x192 fast kernel.
- Consecutive physical CTAs visit `GROUP_M` output rows at a fixed N tile
  before advancing N. This trades a small amount of A reuse for additional B
  reuse in L2.
- `GROUP_M=1` exactly restores the original N-fastest raster.
- The run script accepts `GROUP_M=<positive integer>` and currently defaults
  to 6 after retuning the compact shared-memory swizzle kernel.
- The retained 128x128 fast kernel always uses its original raster: a separate
  large-shape A/B measured 78.398 TFLOPS for the original order versus 77.630
  TFLOPS with GROUP_M=4, so applying the wide-kernel default there would cause
  a 0.98% regression.

## Compact-layout retune (2026-07-25)

GPU: NVIDIA RTX A6000 (SM86). Shape: `8064x10752x8192`, 5 warmups and 20
timed iterations. All configurations used the compact shared-memory swizzle
path and passed validation.

| GROUP_M | Kernel time | Performance |
|---:|---:|---:|
| 1 | 15.676 ms | 90.622 TFLOPS |
| 2 | 13.768 ms | 103.176 TFLOPS |
| 3 | 13.552 ms | 104.822 TFLOPS |
| 4 | 13.418 ms | 105.869 TFLOPS |
| 5 | 13.269 ms | 107.059 TFLOPS |
| 6 | 13.277 ms | 106.994 TFLOPS |
| 7 | 13.271 ms | 107.041 TFLOPS |
| 8 | 13.269 ms | 107.058 TFLOPS |

A longer 10-warmup/50-iteration confirmation over the plateau measured:

| GROUP_M | Kernel time | Performance |
|---:|---:|---:|
| 4 | 13.400 ms | 106.014 TFLOPS |
| 5 | 13.303 ms | 106.784 TFLOPS |
| 6 | 13.264 ms | 107.097 TFLOPS |
| 7 | 13.272 ms | 107.034 TFLOPS |
| 8 | 13.291 ms | 106.884 TFLOPS |

Values 5 through 8 are close, but 6 is the best long-iteration result and is
now the default.  Relative to the former default of 4, it reduces time by
1.01% in the confirmation run.  The search was intentionally stopped at 8.

## Earlier padded-layout benchmark sweep

GPU: NVIDIA RTX A6000 (SM86). Shape: `8064x10752x8192`, 10 warmups and 100
timed iterations. All configurations passed sampled FP32-reference validation.

| GROUP_M | Kernel time | Performance | Registers/thread | Active CTAs/SM |
|---:|---:|---:|---:|---:|
| 1 (original) | 16.216 ms | 87.604 TFLOPS | 118 | 2 |
| 2 | 15.333 ms | 92.648 TFLOPS | 126 | 2 |
| 3 | 15.154 ms | 93.740 TFLOPS | 126 | 2 |
| 4 | 15.177 ms | 93.601 TFLOPS | 126 | 2 |
| 5 | 15.238 ms | 93.228 TFLOPS | 126 | 2 |
| 6 | 15.303 ms | 92.832 TFLOPS | 126 | 2 |
| 7 | 20.121 ms | 70.601 TFLOPS | 126 | 2 |
| 8 | 19.090 ms | 74.414 TFLOPS | 126 | 2 |
| 16 | 28.547 ms | 49.762 TFLOPS | 126 | 2 |
| 32 | 33.149 ms | 42.854 TFLOPS | 126 | 2 |

`GROUP_M=3` and 4 are effectively tied in the initial sweep. Four is the
default because it is the more natural power-of-two grouping and retains the
same two-CTA occupancy. Relative to the original order, the first GROUP_M=4
measurement reduces time by 6.41% and raises throughput by 6.85%.

An interleaved confirmation run (`4 -> 3 -> 1`) measured 15.150, 15.199, and
16.241 ms respectively. In that run GROUP_M=4 raised throughput from 87.466 to
93.766 TFLOPS, a 7.20% improvement.

## Nsight Compute evidence

One launch was replayed for four metric passes. Profiler replay timings are not
normal kernel timings; only the hardware counter comparison is relevant.

| Metric | GROUP_M=1 | GROUP_M=4 | GROUP_M=8 |
|---|---:|---:|---:|
| DRAM bytes read | 11.19 GB | 7.48 GB | 7.14 GB |
| L2 sector hit rate | 60.16% | 73.38% | 74.56% |
| Tensor pipe active | 73.68% | 75.47% | 61.17% |
| Barrier stall | 25.51% | 23.92% | 29.36% |
| Long-scoreboard stall | 9.96% | 8.62% | 17.72% |

GROUP_M=4 cuts DRAM reads by 33.2%, improves L2 hit rate by 13.22 percentage
points, and reduces memory-dependency stalls. GROUP_M=8 obtains slightly more
L2 hits but performs worse because simultaneous reuse becomes a request hot
spot: long-scoreboard stalls almost double and Tensor Core activity falls.

## Reproduction

```bash
GROUP_M=1 ./v14/FP16_GEMM_TensorCore_run.sh 8064 10752 8192 10 100
GROUP_M=4 ./v14/FP16_GEMM_TensorCore_run.sh 8064 10752 8192 10 100
GROUP_M=8 ./v14/FP16_GEMM_TensorCore_run.sh 8064 10752 8192 10 100
```
