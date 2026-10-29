# v8：192x192 CTA、96x96 warp tile

## 保留的生产配置

v8 以 v7 的 XOR shared-memory swizzle、resident-B fragment、四级
`cp.async` pipeline 和 `GROUP_M=6` 为基础，将几何扩大为：

- CTA tile：`192x192x16`；
- warp grid：`2x2`，共 4 warps / 128 threads；
- 每个 warp：`96x96` 输出，即 `6x6` 个 `16x16` fragment；
- Tensor Core 指令：`mma.sync.m16n8k16`；
- 每个 CTA shared memory：48 KiB；
- 输入约束：`M % 192 == 0`、`N % 192 == 0`、`K % 64 == 0`。

只保留完整驻留 B-fragment 的生产路径，不包含 v7 的 B-LDSM batch
消融开关。这样可以避免实验选项改变寄存器布局或指令调度。

## 资源检查（RTX A6000 / SM 8.6）

CUDA 12.8 `nvcc -O3 -arch=sm_86 -Xptxas=-v` 报告：

```text
Used 255 registers, 0 bytes spill stores, 0 bytes spill loads
```

48 KiB shared memory 和 255 registers/thread 均未溢出。按 SM86 的资源
上限，2 个 128-thread CTA 正好需要约 64 Ki registers，因此仍可保持
2 CTA/SM；但寄存器余量很小，后续改动必须重新检查 ptxas 输出。

## 实测

GPU：RTX A6000，`M=8064, N=10752, K=8192`，每次 5 次 warmup、30 次
iteration。v7/v8 交替运行三轮，全部通过 128 点 FP32-reference 校验：

| 版本 | 平均时间 | 性能 |
|---|---:|---:|
| v7 `128x192 / warp 64x96` | 12.665 ms | 112.17 TFLOPS |
| v8 `192x192 / warp 96x96` | 12.025 ms | 118.14 TFLOPS |

v8 相对 v7 平均提升约 **5.3%**（单次运行会受 GPU 时钟和系统负载影响）。
更大的 CTA 减少 M 方向 CTA 数量和 B panel 重复加载，在 occupancy 不变的
情况下提高了 B fragment 的计算复用；代价是寄存器使用逼近硬件上限。

## 运行

```bash
NVCC=/usr/local/cuda-12.8/bin/nvcc \
  ./FP16_GEMM_TensorCore_run.sh 8064 10752 8192 5 30
```

