# CUTLASS FP16 Tensor Core GEMM

该版本与 `cublas/` 使用相同的 row-major 接口、FP16 输入、FP16 累加和
FP16 输出。实现不转置数据，而是用 column-major 视图计算
`C^T[N,M] = B^T[N,K] * A^T[K,M]`。

默认 kernel 使用 CUTLASS 2.x device GEMM API、SM80 Tensor Core、
`256x128x32` threadblock tile、`64x64x32` warp tile 和 3-stage pipeline。

运行基准：

```bash
./cutlass/FP16_GEMM_TensorCore_run.sh 8064 10752 8192 5 20
```

如 CUTLASS 不在默认的 `/home/zhangshuyi/cutlass`，可通过
`CUTLASS_ROOT=/path/to/cutlass` 指定。

## 实测结果

环境：NVIDIA RTX A6000（SM 8.6）、CUDA 12.8、CUTLASS 4.4 源码中的
2.x device GEMM API。矩阵为 `8064x10752x8192`，每轮预热 5 次、计时
20 次。三轮与 cuBLAS 交替测试结果如下：

| 实现 | 平均时间 | 平均性能 |
|---|---:|---:|
| CUTLASS | 13.020 ms | 109.114 TFLOPS |
| cuBLAS | 11.363 ms | 125.019 TFLOPS |

CUTLASS 三轮性能范围为 108.035--110.209 TFLOPS，另一次独立运行达到
110.319 TFLOPS。三轮平均性能约为同期 cuBLAS 的 87.3%。
