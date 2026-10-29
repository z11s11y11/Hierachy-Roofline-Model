# Compact shared-memory swizzle result

## Change

- Replaced the default eight-half row padding with a compact XOR-swizzled
  layout inside each `16x16` half tile.
- Kept every global-to-shared transfer as an aligned 16-byte `cp.async`.
- Replaced the fast paths' WMMA operand loads with explicit
  `ldmatrix.x4` / `ldmatrix.x4.trans` and `mma.m16n8k16` instructions, so the
  consumer applies the same non-affine layout as the producer.
- Retained the original padded WMMA path as a compile-time A/B baseline:

  ```bash
  SWIZZLE=0 ./v20/FP16_GEMM_TensorCore_run.sh M N K WARMUP ITERS
  ```

## Layout

Each `16x16` tile contains 32 vectors of eight half elements.  For local row
`r` and eight-half vector `v`, the shared vector slot is:

```cpp
logical = r * 2 + v;
physical = logical ^ (r >> 2);
```

The mapping preserves 16-byte alignment and permutes, rather than expands,
the tile storage.  For each eight-row group consumed by `ldmatrix`, vector
starts cover all eight 16-byte shared-memory bank groups.

## Correctness

All sampled FP32-reference validations passed on NVIDIA RTX A6000 (SM86):

| Shape | Path | Result |
|---|---|---|
| `128x128x64` | 128x128 swizzle | PASS |
| `128x192x64` | 128x192 swizzle | PASS |
| `256x384x256` | multi-CTA 128x192 swizzle | PASS |
| `257x259x263` | boundary-safe fallback | PASS |
| `8064x10752x8192` | benchmark wide path | PASS |
| `8192x10240x8192` | benchmark 128x128 path | PASS |

## Resources

CUDA 12.8, `-O3 -arch=sm_86`:

| Kernel | Layout | Registers/thread | Shared/CTA | Active CTAs/SM | Spill |
|---|---|---:|---:|---:|---:|
| 128x128 | padding | 92 | 41 KiB | 2 | 0 B |
| 128x128 | swizzle | 92 | 32 KiB | 2 | 0 B |
| 128x192 | padding | 126 | 49 KiB | 2 | 0 B |
| 128x192 | swizzle | 126 | 40 KiB | 2 | 0 B |

Swizzling saves 9 KiB per CTA without changing occupancy or register use.

## Performance

Five warmups and 20 timed iterations:

| Shape / kernel | Layout | Average | Performance | Validation |
|---|---|---:|---:|---|
| `8064x10752x8192`, 128x192 | padding | 15.114 ms | 93.991 TFLOPS | PASS |
| `8064x10752x8192`, 128x192 | swizzle | 13.421 ms | 105.844 TFLOPS | PASS |
| `8064x10752x8192`, 128x192 repeat | swizzle | 13.381 ms | 106.162 TFLOPS | PASS |
| `8192x10240x8192`, 128x128 | padding | 17.351 ms | 79.209 TFLOPS | PASS |
| `8192x10240x8192`, 128x128 | swizzle | 15.344 ms | 89.571 TFLOPS | PASS |

The wide path is 11.2%-11.5% faster by elapsed time.  The 128x128 path is
11.6% faster.

## Nsight Compute shared-load counters

The personal Nsight Compute 2025.1 wrapper was used with its private temporary
directory.  The profiled shape was `128x192x8192`, one CTA, after skipping one
warm-up launch:

| Layout | Shared-load bank conflicts | Shared-load wavefronts | NCU duration |
|---|---:|---:|---:|
| padding | 0 | 134,604 | 493.120 us |
| swizzle | 0 | 132,601 | 482.688 us |

Both layouts eliminate bank conflicts.  Swizzling reduces measured shared-load
wavefronts by 1.49%; therefore the full-grid 11%-plus speedup should not be
described as removing residual conflicts from the padded kernel.  It combines
the compact layout, the explicit `ldmatrix`/`mma.sync` path, and their effects
on the generated instruction/address pipeline.

Raw reports:

- `.ncu/reports/v20_padding_shared_load.ncu-rep`
- `.ncu/reports/v20_swizzle_shared_load.ncu-rep`
