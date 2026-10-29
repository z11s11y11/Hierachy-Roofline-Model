# v6：紧凑共享内存 swizzle

## 保留的算子

只保留 `128x192`、`GROUP_M=6`、XOR-swizzled 共享内存布局的四级异步核。尺寸要求为 `M % 128 == 0`、`N % 192 == 0`、`K % 64 == 0`。

## 相比最初 WMMA 基线的优化

- 延续 Tensor Core、宽 tile、CTA 分组及 `cp.async` 流水。
- 把每个 `16x16` half tile 看作 32 个 16-byte vector，以 `physical = logical ^ (row >> 2)` 重排，取代每行 8 个 half 的 padding。
- 显式使用 `ldmatrix.x4`、`ldmatrix.x4.trans` 和 `mma.m16n8k16`，让加载端与非仿射共享内存布局匹配。
- 每 CTA 共享内存由 49 KiB 降为 40 KiB，同时保持 16-byte 对齐和 2 CTA/SM。

历史测试中宽核由约 15.1 ms 降到 13.4 ms，提升约 11%；最终日志记录 106.311 TFLOPS。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh 1024 960 1024 5 20
```
