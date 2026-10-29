# v5：L2 友好的 CTA 排布

## 保留的算子

只保留 `128x192` 四级异步核，固定采用测得较优的 `GROUP_M=4` CTA raster。尺寸要求为 `M % 128 == 0`、`N % 192 == 0`、`K % 64 == 0`。

## 相比最初 WMMA 基线的优化

除 Tensor Core、大 tile、寄存器累加及 `cp.async` 流水外，本版重新映射物理 CTA 到输出 tile：相邻 CTA 先遍历 4 个 M tile，再前进 N。这样相邻 CTA 更可能复用同一个 B panel 的 L2 cache line。

Nsight Compute 历史数据表明，相比原 N-fastest 顺序，`GROUP_M=4` 将 DRAM 读取从 11.19 GB 降到 7.48 GB，L2 sector hit rate 从 60.16% 提高到 73.38%；确认测试的吞吐提升约 7.2%。过大的 group 会形成请求热点，因此没有保留其它 group 变体。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh 1024 960 1024 5 20
```
