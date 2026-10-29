# v4：128x192 宽 tile

##

尺寸必须满足 `M % 128 == 0`、`N % 192 == 0`、`K % 64 == 0`。

## 相比最初 WMMA 基线的优化

在 v3 异步流水基础上，将 CTA 的 N 方向从 128 扩到 192。每个 warp 的 N fragment 从 4 个增到 6 个，同一份 A panel 可参与更多 MMA，降低 A 的重复读取及每个输出元素分摊到的同步/调度开销。

四级环形缓冲使用 49 KiB 动态共享内存，并通过 CUDA function attribute 申请较大的单 CTA 共享内存；在 SM86 上仍保持 2 CTA/SM。历史 A/B 测试相对 `128x128` 将 18.397 ms 降至 16.478 ms，吞吐提高约 11.6%。

## 运行

```bash
./FP16_GEMM_TensorCore_run.sh 8064 10752 8192 5 20
```
