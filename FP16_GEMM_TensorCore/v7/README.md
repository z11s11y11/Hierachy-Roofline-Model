# v7：4-warp、每 warp 64x96 几何

## 保留的算子

固定配置为：`128x192` CTA、`2x2` warp 网格（4 warp）、每 warp 负责 `64x96` 输出、`GROUP_M=6`、XOR swizzle，并让 B fragments 完整驻留寄存器。尺寸要求为 `M % 128 == 0`、`N % 192 == 0`、`K % 64 == 0`。

## 相比最初 WMMA 基线的优化

本版包含前代的 Tensor Core、宽 tile、四级 `cp.async`、L2 CTA 分组和紧凑共享内存布局，并进一步定型寄存器/并行度平衡：

- 选择 4 warp/CTA，每个 warp 沿 M 方向持有 4 个、沿 N 方向持有 6 个 16x16 fragment。
- 线程数降为 128 后，每个线程通过步进循环搬运两个 A 的 16-byte 向量，确保完整覆盖 128x16 A panel。
- 不采用早拷贝的 B streaming 实验；让 B fragments 驻留寄存器，避免破坏 HMMA 指令流水。
- 每线程使用 226 个寄存器、无 spill；40 KiB shared memory 下可驻留 2 CTA/SM，即 8 warp/SM。

RTX A6000（SM 8.6）上以 `M=8064, N=10752, K=8192`、3 次 warmup、20 次迭代实测为 12.641 ms、112.376 TFLOPS，128 个随机/边界采样点验证通过。

`mlsys_profile/` 保存了同一尺寸、5 次 warmup 后单次 kernel 的 NCU PM Sampling 数据：

- `v7_pipeline_timeline.svg`：0--11 ms 截取范围的四面板时间线；
- `v7_pm_samples.csv`：图中 7 条曲线的原始 10 us 采样；
- `v7_pm_timeline.ncu-rep`：Nsight Compute 原始报告；
- `v7_timeline_summary.txt`：kernel 名称和时间轴摘要。

`B_LDSM_HMMA_PIPELINE_RESULT.md` 记录了 B fragment 的 1B/2B/3B
`LDSM`/HMMA 依赖链实验。三种流水深度均未超过完整驻留基线，因此默认仍为
`FP16_GEMM_B_LDSM_BATCH=0`。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh 1024 960 1024 5 20
```
