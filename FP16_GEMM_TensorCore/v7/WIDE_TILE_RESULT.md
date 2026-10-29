# v21 register/shared-memory balance experiment

## Baseline

GPU: NVIDIA RTX A6000 (SM86), shape `8064x10752x8192`.

The selected `128x192`, 8-warp kernel originally used 126 registers/thread and 40 KiB
shared memory/CTA. Two CTAs occupy all 65,536 SM registers after allocation
rounding, but only 80 of the available 100 KiB shared memory. It sustains
106.392 TFLOPS in the 5-warmup/20-iteration comparison.

## Wider-N experiment

The experimental early-copy inline-B path moved future `cp.async` operations
ahead of operand loads and reduced compiler-reported registers without stack or
spill:

| CTA tile | Registers/thread | Shared/CTA | CTAs/SM | Performance |
|---|---:|---:|---:|---:|
| 128x192 baseline | 126 | 40 KiB | 2 | 106.392 TFLOPS |
| 128x192 early-copy | 118 | 40 KiB | 2 | 90.017 TFLOPS |
| 128x224 early-copy | 125 | 44 KiB | 2 | 102.741 TFLOPS |
| 128x256 early-copy | 128 | 48 KiB | 2 | 95.843 TFLOPS |

All variants passed validation. SASS inspection showed that CUDA 12.8 ptxas
still hoists all independent B `LDSM` instructions before the HMMA strip. The
register reduction therefore comes mainly from moving future-copy address
bookkeeping, not from a true four-register streamed B fragment. The wider tiles
consume the spare shared memory but reduce throughput, so they are not selected.

## More-warps experiment

The second approach kept the `128x192` CTA tile and divided it among more warps,
reducing each warp's accumulator footprint:

| Warp grid | Threads/CTA | Registers/thread | Spill | CTAs/SM | Warps/SM | Performance |
|---|---:|---:|---:|---:|---:|---:|
| 4x2 baseline | 256 | 126 | 0 B | 2 | 16 | 106.392 TFLOPS |
| 4x3 | 384 | 79 | 0 B | 2 | 24 | 92.411 TFLOPS |
| 4x4 | 512 | 64 | 4 B load/store | 2 | 32 | 94.805 TFLOPS |

Higher occupancy is not beneficial here. More warp columns replicate A
`ldmatrix` work and increase warp scheduling and barrier pressure. The baseline
already reaches nearly 90% HMMA-pipeline utilization, while long-scoreboard
stalls are only about 2.6%, so additional latency hiding has little headroom.

## Selected configuration

- `FP16_GEMM_WIDE_BLOCK_N=192`
- `FP16_GEMM_WIDE_WARP_COLS=2` (8 warps/CTA)
- `FP16_GEMM_STREAM_B_FRAGMENTS=0` (fully resident B fragments)
- `FP16_GEMM_GROUP_M=6`

After making the copy distribution geometry-generic and rebuilding the selected
default, ptxas reports 124 registers/thread, 0 stack, and 0 spill. The final
5-warmup/30-iteration validation run measured 13.062 ms / 108.752 TFLOPS.
A controlled CUDA 12.8 A/B measured the old 126-register binary at 13.360 ms /
106.333 TFLOPS and the final 124-register binary at 13.047 ms / 108.882 TFLOPS,
a 2.4% throughput increase. Final NCU measurements report 89.57% HMMA-pipeline
utilization, 87.59% L2 hit rate, 2.34 Gbyte DRAM reads, and 2.43%
long-scoreboard stalls.

The extra shared-memory capacity cannot replace live FP16 accumulator registers
on Ampere without extra shared-memory traffic. For a future redesign, the most
promising direction is a true dependency-chained B-load/HMMA schedule or a
different MMA tiling that reduces operand-load duplication—not manually spilling
accumulators to shared memory.
