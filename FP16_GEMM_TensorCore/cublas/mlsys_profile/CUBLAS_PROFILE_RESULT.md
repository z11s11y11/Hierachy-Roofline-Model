# cuBLAS Tensor Core PM Sampling 结果

## 配置

- GPU：NVIDIA RTX A6000，SM 8.6，84 SM
- GEMM：`M=8064, N=10752, K=8192`
- 输入、累加器、输出：FP16 / FP16 / FP16
- API：`cublasGemmEx` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- 实际 kernel：`ampere_h16816gemm_256x128_ldg8_stages_32x3_nn`
- launch：grid `(42,63,1)`，block `(256,1,1)`，共 2646 CTA
- 资源：154 registers/thread（160 allocated），74.752 KiB shared/CTA
- occupancy 限制：register 和 shared memory 均限制为 1 CTA/SM
- PM Sampling：10 us，四个 replay pass groups，与 v7 相同的 7 条曲线

## 性能

三轮 cuBLAS/v7 交替测试，每轮 3 warmups、30 iterations：

| 实现 | 平均时间 | 平均性能 | 相对 v7 |
|---|---:|---:|---:|
| cuBLAS | 11.454 ms | 124.019 TFLOPS | +13.1% |
| v7 | 12.949 ms | 109.704 TFLOPS | 基准 |

## Tensor Core 指标

独立 NCU counter profile：

| 指标 | cuBLAS |
|---|---:|
| HMMA Tensor Active，elapsed | 96.37% |
| HMMA Tensor Active，SM active | 97.87% |
| HMMA 动态指令 | 346,816,512 |
| L2 hit rate | 90.17% |
| DRAM read | 2.16 GB |
| DRAM write | 172.93 MB |
| barrier stall | 0.30% |
| long-scoreboard stall | 0.28% |
| math-pipe throttle | 70.52% |

HMMA 指令数与 v7 完全相同，但 cuBLAS 的 Tensor pipeline 填充率更高。
很高的 math-pipe throttle 表示大量 warp 已经准备好，只是在等待 Tensor
pipeline，而不是等待 global memory。

## 时间线形状

- 稳定平台的 Tensor Active 中位数约为 **99.32%**，v7 约为 92.99%。
- 整个 active window 的 PM Sampling 平均值约为 **96.25%**。
- 图中仍能看到约 32 个很窄的周期性凹口，对应 CTA wave 边界。
- 2646 CTA / 84 SM = 31.5 waves。由于只能驻留 1 CTA/SM，最后半个
  wave 只有约一半 SM 工作，因此尾部出现一段约 50% Tensor Active。
- 除首尾和 wave 边界外，Tensor 曲线几乎贴着 100%，明显比 v7 平坦。

生成文件：

- `cublas_pipeline_timeline.svg`：与 v7 相同版式的图；
- `cublas_pm_samples.csv`：7 条原始采样序列；
- `cublas_pm_timeline.ncu-rep`：PM Sampling 原始报告；
- `cublas_kernel_metrics.ncu-rep`：独立硬件计数器报告；
- `cublas_timeline_summary.txt`：kernel 和时间轴摘要。
