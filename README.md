# Hierachy-Roofline-Model
Hierachy Roofline Model，实验要求见 [methods.md](methods.md)。

本项目使用 ERT 测量 GPU、用 Nsight Compute 采集 v1–v8 / cuBLAS / CUTLASS 的 L1、L2、DRAM 流量，并绘制分层 Roofline。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt
.venv/bin/python scripts/run_experiment.py all --device 0 --cutlass-root ../cutlass
```

需要 NVIDIA GPU、CUDA Toolkit（nvcc）、Nsight Compute（ncu）、GPU 性能计数器访问权限以及 CUTLASS 头文件。脚本自动检测 GPU 型号与 SM 架构，分别保存到 `resluts/A6000/`、`resluts/H100/` 等目录。

- [ERT 运行说明](run_ERT.md)
- [10 个算子测试与绘图说明](run_operator.md)
- [A6000 实测报告](resluts/A6000.md)
- [A6000 Roofline 图片](resluts/A6000/roofline.png)

已有 A6000 结果来自共享 GPU 环境，报告保留了原始数据和测量条件。A6000 使用 GDDR6，图中以 DRAM 表示显存层。
