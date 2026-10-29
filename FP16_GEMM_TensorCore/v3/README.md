# v3：四级异步流水

尺寸必须满足 `M % 128 == 0`、`N % 128 == 0`、`K % 64 == 0`；不满足时接口返回 `cudaErrorInvalidValue`。

## 相比最初 WMMA 基线的优化

- CTA tile 从 `32x32` 扩大到 `128x128`，每次加载的 A/B panel 被更多 MMA 复用。
- 使用 8 个 warp，每个 warp 负责 `2x4` 个 WMMA fragment，C tile 始终驻留寄存器。
- 用 `cp.async` 将全局内存到共享内存的拷贝与当前 tile 的 Tensor Core 计算重叠。
- A、B 各使用 4 个 `K=16` 环形 stage，并保持最多两个待完成异步组，隐藏 v2 定位出的显存延迟。
- 共享内存行加入 8 个 half 的 skew，避免 `16/128` leading dimension 导致的 bank 冲突。

历史日志在 RTX A6000、`8192x10752x8192` 上记录为 78.399 TFLOPS。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh 8192 10752 8192 5 20
```
