# Hierachy-Roofline-Model

按 [methods.md](methods.md) 测量 GPU 的 FP32、稠密 FP16 Tensor Core 算力，以及 L1TEX、L2、显存带宽；分析 v1–v8、cuBLAS、CUTLASS 共 10 个 GEMM 并绘制分层 Roofline。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt
.venv/bin/python scripts/run_experiment.py all --device 0 --cutlass-root ../cutlass
```

需要 CUDA Toolkit、Nsight Compute、CUTLASS 头文件和可访问的 GPU 性能计数器。详见 [硬件测量](run_ERT.md) 与 [算子测量和绘图](run_operator.md)。

本次在物理 GPU 4 的 RTX A6000 上运行，结果见 [resluts/A6000.md](resluts/A6000.md)。原始日志、CSV/JSON 和图片位于 `resluts/A6000/`。实测来自共享 GPU 环境。

A6000 使用 GDDR6，结果中的 DRAM 对应任务所称的 HBM 层。图中实线表示实测值，虚线表示本地 NVIDIA GA102 白皮书 Table 3 的理论值：FP32 38.7 TFLOP/s、稠密 FP16 Tensor 154.8 TFLOP/s、显存 768 GB/s。
