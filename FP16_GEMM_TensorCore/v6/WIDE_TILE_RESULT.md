# v13 128x192 wide-tile result

## Change

- Added a `128x192` SM86 fast kernel while retaining the original `128x128`
  kernel and the boundary-safe generic kernel.
- The wide kernel uses a four-stage 49-KiB dynamic-shared-memory ring and
  opts in to the SM86 larger per-block shared-memory limit.
- Dispatch selects `128x192` when `M % 128 == 0`, `N % 192 == 0`, and
  `K % 64 == 0`; otherwise it selects the existing paths.

## Resources

| Kernel | Registers/thread | Shared/CTA | Active CTAs/SM | Warps/SM | Spill |
|---|---:|---:|---:|---:|---:|
| 128x128 baseline | 93 | 41 KiB | 2 | 16 | 0 B |
| 128x192 wide | 118 | 49 KiB | 2 | 16 | 0 B |

The wide kernel consumes 98 of the available 100 KiB shared memory per SM.

## A/B benchmark

GPU: NVIDIA RTX A6000 (SM86), shape `8192x10752x8192`, 10 warmups and 100
timed iterations.

| Kernel | Average | Performance | Validation |
|---|---:|---:|---:|
| 128x128 baseline | 18.397 ms | 78.444 TFLOPS | PASS |
| 128x192 wide | 16.478 ms | 87.579 TFLOPS | PASS |

The wide tile reduces kernel time by 10.43% and increases throughput by
11.64%.

Additional validation passed for the retained `128x128` path at
`256x256x256` and the generic path at `257x259x263`.
