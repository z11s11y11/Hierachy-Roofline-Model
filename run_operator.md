# 10 个 FP16 GEMM 算子与分层 Roofline

先按 [run_ERT.md](run_ERT.md) 完成硬件测量；也可一次运行全部实验：

```bash
.venv/bin/python scripts/run_experiment.py all --device 0 --cutlass-root ../cutlass
```

三个依赖代码仓库已经位于当前项目。CUTLASS 算子还需要 NVIDIA CUTLASS 头文件，默认查找 `../cutlass/include/cutlass/cutlass.h`。若未安装，可以克隆后指定路径：

```bash
git clone --branch v3.9.2 --depth 1 https://github.com/NVIDIA/cutlass.git ../cutlass
export CUTLASS_ROOT="$PWD/../cutlass"
```

分别执行算子测量和离线画图：

```bash
.venv/bin/python scripts/run_experiment.py operators --device 0 --cutlass-root ../cutlass
.venv/bin/python scripts/run_experiment.py plot --output resluts/A6000
```

离线绘图只需要 Python 依赖和已保存的数据，不需要 GPU、nvcc 或 ncu。

统一测试 `v1`–`v8`、`cublas`、`cutlass` 共 10 个实现。编译真实算子源文件，搭配 `scripts/operator_runner.cu`，不会调用仓库里的历史测试二进制或使用历史 result.log。编译命令和日志保存在 `raw/operators/*_compile.log*`。

默认 M=N=K=3072，各算子使用相同输入、相同随机种子和 `C=A*B`。M 必须为 384 的倍数、N 为 768 的倍数、K 为 64 的倍数，以兼容全部 CTA tile。调整尺寸示例：

```bash
.venv/bin/python scripts/run_experiment.py operators --device 0 \
  --m 6144 --n 6144 --k 4096 --warmup 5 --iterations 20 --repeats 3 \
  --cutlass-root ../cutlass --output resluts/A6000_large
```

输入、输出和已有算子的累加类型均沿用源实现。测试框架对四个角点及 124 个固定随机输出位置，用 CPU FP64 累加计算参考值；每个位置要求 `abs(error) <= 0.02 + 0.03*abs(reference)`，采样整体相对 L2 误差还需不超过 3%。失败立即退出，不把错误结果写成有效采样。所有采样误差随结果保存。

计时与 profiler 分开运行，默认每个算子 3 轮：

1. CUDA Event：预热 3 次，连续运行 10 次求单次平均时间；3 轮取中位数。
2. ncu：仅采集 `roofline` NVTX 范围内的一次完整 GEMM 调用。若库实现有多个辅助 kernel，会累加全部 kernel 的流量与时间。排除初始化、预热、普通计时和 CPU 验证。3 轮取总 kernel 时间的中位样本，性能与各层流量均来自该样本。

原始 ncu 指标如下：

```text
gpu__time_duration.sum
l1tex__t_sectors.sum
lts__t_sectors.sum
dram__bytes.sum
```

计算规则：

```text
FLOP = 2*M*N*K                         # 有效算法工作量，FMA=2 FLOP
L1_bytes = l1tex__t_sectors.sum * 32
L2_bytes = lts__t_sectors.sum * 32
DRAM_bytes = dram__bytes.sum
AI_level = FLOP / level_bytes          # FLOP/byte
TFLOP/s = FLOP / kernel_seconds / 1e12
```

这里 L1 表示 L1TEX 请求接口，并非 L1 hit 流量；`cp.async.cg` 绕过 L1 缓存仍会产生接口请求。共享内存内部流量不计入该 L1 定义。FLOP 使用有效 GEMM 工作量，不包含实现的填充、重复运算或地址指令。ncu 每次 replay 清缓存，Event 计时允许缓存预热，两类性能不应混用。

结果位置：

| 文件 | 内容 |
|---|---|
| `resluts/A6000.md` | 硬件上限、10 个算子的算力与 3 层 AI、测量条件 |
| `resluts/A6000/hardware.json` | 硬件、ERT 引用、屋顶与全部校准样本 |
| `resluts/A6000/operators.csv` | 10 行汇总，含实测字节数和时间 |
| `resluts/A6000/operators.json` | 全部重复样本、kernel 名和误差 |
| `resluts/A6000/raw/` | GPU 环境快照、编译及采集日志、原始命令 |
| `resluts/A6000/roofline.png` | ERT 配合补充校准、10 个算子的分层 Roofline |
| `resluts/A6000/roofline.svg` / `.pdf` | 可编辑 / 可导出的矢量图 |
| `resluts/A6000/ert_working_set.png` | 原始 ERT 工作集带宽和 FP32 算力 |

绘图改造自 `nersc-roofline/Plotting/plot_roofline_hierarchical.py`，已支持 Python 3、自动坐标范围、10 个 marker 和 JSON 输入，仍可读取上游文本示例：

```bash
.venv/bin/python nersc-roofline/Plotting/plot_roofline_hierarchical.py \
  nersc-roofline/Plotting/data_hierarchical.txt --output .cache/legacy_roofline
```

CPU 端解析回归检查：

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```
