# v1：WMMA 基线

## 保留的算子

本目录只保留 `fp16_gemm_tensor_core_kernel`。输入、累加器和输出均为 FP16，矩阵采用 row-major 布局；支持任意正整数 `M/N/K`，边界 tile 由零填充和带条件的写回处理。

## 相比朴素矩阵乘的优化

- 用 `16x16x16` WMMA 指令在 Tensor Core 上代替逐元素 CUDA Core 乘加。
- 一个 CTA 含 4 个 warp，计算一个 `32x32` 的 C tile，提高线程块级并行度。
- A、B 的 `K=16` panel 先协作搬入共享内存，减少同一数据被各 warp 重复读取。
- C 的 WMMA fragment 在整个 K 循环中留在寄存器，循环结束后才写回显存。

这是后续版本的性能基线，而不是朴素的一线程一元素实现。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh 2048 2048 2688 10 100
```

可通过 `NVCC=/path/to/nvcc` 和 `CUDA_ARCH=sm_86` 指定编译器及架构。
