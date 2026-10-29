# ERT 与 GPU 屋顶校准

在仓库根目录执行，Linux 环境需要 CUDA Toolkit、Nsight Compute 和 Python 3。默认优先使用 PATH 或 `/usr/local/cuda*/bin/` 中的工具，也可通过 `NVCC` / `NCU` 或 `--nvcc` / `--ncu` 指定。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt
.venv/bin/python scripts/run_experiment.py ert --device 0
```

`--device` 为一张物理 GPU 的 ID 或 UUID；也可设置 `CUDA_VISIBLE_DEVICES`。程序通过 CUDA 查询真实型号和 SM 架构，A6000 输出至 `resluts/A6000/`，H100 输出至 `resluts/H100/`。自定义目录使用 `--output resluts/my_run`；对应 Markdown 报告会生成在 `resluts/my_run.md`。

本步骤实际编译、运行 `cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/ert` 的 `driver1` / `kernel1`，并执行上游 Python 后处理，生成原生 `roofline.json`。生成的完整配置在 `ert_config.txt`，命令记录在 `raw/ert.log.command.json`。默认参数如下：

- FP32，FLOP/element = 1、2、64、1024。
- GPU blocks = SM 数和 4 倍 SM 数，threads/block = 256。
- 最大工作集 64 MiB，从 8192 个 FP32 元素开始，每次翻倍。
- 每种配置重复 3 次；可用 `--repeats 5 --ert-memory-mib 256` 增加采样。

每次执行创建新的 ERT 原始运行目录，不复用旧测量。`ert_samples.csv` 汇总工作集、重复访问次数、时间、logical bytes 和 FLOP；`ert/<时间>/Run.001/` 保留原始 `try.*`、后处理文件和原生 `roofline.json`。

不能把 ERT 标量 FP32 算力当作 FP16 Tensor Core 上限，也不能直接把 ERT logical read/write bandwidth 当作 L1 带宽。因此脚本另编译 `scripts/calibrate.cu`，补充实测：

| 校准 | 工作集 / 指令 | 屋顶计算方式 |
|---|---|---|
| L1TEX | 每 CTA 4 KiB，重复 `.ca` 读取 | `l1tex__t_sectors.sum * 32 / seconds` |
| L2 | 一半 L2 容量，重复 `.cg` 读取，绕过 L1 | `lts__t_sectors.sum * 32 / seconds` |
| DRAM | max(64 MiB, 16 × L2)，`.cg` 完整扫描 | `dram__bytes.sum / seconds` |
| FP16 Tensor Core | 每 warp 8 个独立 WMMA FP16 累加器 | `warps * 8 * repeats * 8192 / seconds` |

`seconds` 来自同一 profile 的 `gpu__time_duration.sum`；GB/s 使用十进制 10⁹，TFLOP/s 使用 10¹²。校准在 profiler 的首次 replay 前清缓存，统计整个 kernel，包含冷启动开销。读探针校验所有输出，Tensor 探针校验所有输出元素；Tensor 输入和迭代数确保期望值在 FP16 下可精确表示。各屋顶取 3 次实测的最大值，完整样本保存在 `calibration_samples.json` 和 `hardware.json`。

运行不会调整 GPU 时钟；ncu 使用 `--clock-control none --cache-control all`。当前测量发生在共享机器，报告中的实测上限是该环境的观察值；更稳定的机器峰值需要在空闲 GPU 上重跑。本脚本使用仓库内 `.cache/ncu-tmp` 避免共享 `/tmp` 的锁文件权限冲突。若 ncu 报性能计数器权限不足，需由管理员配置允许访问后重跑，程序会保存错误日志并退出，不生成替代测量。

当前上游 ERT 已有模板化 FP32 内核，配置 `ERT_PRECISION FP32` 即可启用，因此不运行 FP64 测量。`scripts/calibrate.cu` 补充了原 ERT 缺少的稠密 FP16 Tensor Core 和缓存加载内核。ERT 源码来自 Bitbucket commit `09663aad764d3c6cefa4d9de1f1213b5f03af9df`，已作为普通源码目录保存，保留 Copyright / License。实验脚本复用并修订自相邻旧实验目录，当前结果全部重新采集，`sources.json` 保存源文件 SHA256。

图中的 FP32 值取原生 ERT `roofline.json` 的汇总值（重复实验取中位时间，再选平台峰值）；原始单次样本最高值单独保存在 `hardware.json` 的 `ert_fp32_max_sample_TFLOPS`，避免混淆两种口径。ERT 原生缓存层识别不用于本图的物理带宽屋顶，后者来自接口计数器校准。

三个依赖已按普通源码目录组织，原先的 ERT gitlink 已移除；现有两个依赖的 `.git` 元数据备份至本地忽略目录 `.cache/source-git/`，上游版本见 `scripts/upstream_sources.json`。后续上传可以包含依赖源码，无需再次删除嵌套 Git 目录。

本次命令为 `.venv/bin/python scripts/run_experiment.py all --device 4 --cutlass-root /home/zhangshuyi/cutlass`。理论值取本地白皮书 Table 3（印刷页 15–16）：FP32 38.7 TFLOP/s、稠密 FP16 Tensor Core 154.8 TFLOP/s、GDDR6 768 GB/s；虚线展示理论值，实线展示实测值。309.6 TFLOP/s 是 2:4 稀疏算力，不适用于本实验的稠密 GEMM。白皮书基于 1800 MHz Boost，未锁频实测可能略超出标称值。

然后执行 [算子测试与绘图](run_operator.md)。
