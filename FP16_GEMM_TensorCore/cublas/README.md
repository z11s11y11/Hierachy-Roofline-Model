# cuBLAS FP16 Tensor Core GEMM

该版本用 `cublasGemmEx` 计算与 v7 相同的 row-major
`C[M,N] = A[M,K] * B[K,N]`。数据类型、累加器和输出均为 FP16，算法使用
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`。

运行基准：

```bash
./cublas/FP16_GEMM_TensorCore_run.sh 8064 10752 8192 5 20
```

`mlsys_profile/` 包含与 v7 相同指标和版式的 10 us PM Sampling 时间线。

本机 RTX A6000、`8064×10752×8192` 的三轮交替测试平均结果：

| 实现 | 时间 | 性能 |
|---|---:|---:|
| cuBLAS | 11.454 ms | 124.019 TFLOPS |
| v7 | 12.949 ms | 109.704 TFLOPS |

cuBLAS 比 v7 快约 13.1%。详细 profile 分析见
`mlsys_profile/CUBLAS_PROFILE_RESULT.md`。
