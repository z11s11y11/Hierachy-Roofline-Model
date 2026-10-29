# v2：64×64 双缓冲 WMMA

本目录保留一个 `fp16_gemm_tensor_core_kernel`。输入、FP16 累加器和输出均为 FP16，矩阵采用 row-major 布局。算子带有补零加载和边界写回路径，因此支持任意正整数 `M/N/K`。

## 相比 v1 的优化

- CTA tile 从 `32x32` 扩大到 `64x64`，线程数从 4 个 warp 增加到 8 个 warp，提高 A/B panel 的数据复用率。
- 每个 warp 复用一个 A fragment，与两个相邻的 B fragment 计算两个 `16x16` 输出 fragment。
- B 在共享内存中保持 row-major，使 global memory 的 N 方向访问连续。
- A/B 使用两个 `K=16` shared-memory stage 进行乒乓缓冲，总计占用 8 KiB。
- 在 SM80+、地址对齐且 tile 完整时，用 16-byte `cp.async` 预取下一个 K tile，使数据搬运与当前 WMMA 计算部分重叠。
- 完整输出 tile 直接从 accumulator 写回 global memory；只有边界 tile 才复用流水缓冲完成安全写回。
- SM70/75、非对齐尺寸及边界 tile 自动使用带越界检查的同步加载路径。

## 历史结果

`result.log` 记录的环境为 RTX A6000，形状 `2304x2688x2048`，10 次预热、100 次计时：平均 1.176 ms，21.570 TFLOPS，验证通过。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh
./FP16_GEMM_TensorCore_run.sh 2304 2688 2048 10 100
```

可通过 `NVCC=/path/to/nvcc` 和 `CUDA_ARCH=sm_86` 指定编译器及目标架构。
