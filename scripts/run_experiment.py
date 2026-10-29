#!/usr/bin/env python3
"""Run real ERT, counter-based GPU calibration, and ten FP16 GEMMs."""
import argparse
import csv
import hashlib
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ERT = ROOT / "cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0"
VARIANTS = [f"v{i}" for i in range(1, 9)] + ["cublas", "cutlass"]
METRICS = ["gpu__time_duration.sum", "l1tex__t_sectors.sum",
           "lts__t_sectors.sum", "dram__bytes.sum"]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def executable(name, override=None):
    candidates = [override, shutil.which(name), f"/usr/local/cuda/bin/{name}"]
    candidates += [str(p) for p in sorted(Path("/usr/local").glob(f"cuda-*/bin/{name}"), reverse=True)]
    candidates += [str(p) for p in sorted(Path("/usr/local").glob(f"cuda-*/nsight-compute*/{name}"), reverse=True)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError(f"找不到 {name}；请设置 {name.upper()} 或对应命令行参数")


def run(command, log, env, timeout=600):
    command = [str(c) for c in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)
    with log.with_suffix(log.suffix + ".command.json").open("w") as f:
        json.dump({"argv": command, "cwd": str(ROOT),
                   "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                   "TMPDIR": env.get("TMPDIR"),
                   "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    with log.open("w") as f:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=f,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"命令超时；原始记录：{log}") from error
    output = log.read_text(errors="replace")
    if result.returncode:
        raise RuntimeError(f"命令失败 ({result.returncode})：{log}\n{output[-4000:]}")
    return output


def json_line(text):
    return json.loads(next(line for line in reversed(text.splitlines()) if line.startswith("{")))


def parse_ncu(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID","')), None)
    if start is None:
        raise RuntimeError("ncu 没有导出计数器 CSV；请检查原始日志")
    kernels = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        metric = row.get("Metric Name")
        if metric not in METRICS:
            continue
        key = row["ID"]
        record = kernels.setdefault(key, {"kernel": row["Kernel Name"], "metrics": {}})
        value = float(row["Metric Value"].replace(",", ""))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"无效计数器 {metric}: {value}")
        unit = row["Metric Unit"]
        if metric == "gpu__time_duration.sum":
            value *= {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[unit]
        elif metric != "dram__bytes.sum":
            if unit != "sector":
                raise RuntimeError(f"意外的 sector 单位：{unit}")
            value *= 32  # NVIDIA L1TEX and L2 sectors are 32 bytes.
        elif unit != "byte":
            raise RuntimeError(f"意外的 DRAM 单位：{unit}")
        record["metrics"][metric] = value
    if not kernels or any(set(k["metrics"]) != set(METRICS) for k in kernels.values()):
        raise RuntimeError("ncu 指标缺失；不会用理论访存量替代实测值")
    totals = {metric: sum(k["metrics"][metric] for k in kernels.values()) for metric in METRICS}
    if totals[METRICS[0]] <= 0:
        raise RuntimeError("ncu kernel 时间无效")
    return {"seconds": totals[METRICS[0]], "L1_bytes": totals[METRICS[1]],
            "L2_bytes": totals[METRICS[2]], "DRAM_bytes": totals[METRICS[3]],
            "kernels": list(kernels.values())}


def profile(ncu, binary, arguments, log, env, skip=1):
    # Profile all kernels in one final GEMM call, including possible library
    # helper kernels. Warmups end before a dedicated NVTX range (operator runner).
    command = [ncu, "--csv", "--clock-control", "none", "--cache-control", "all",
               "--metrics", ",".join(METRICS)]
    if skip is None:
        command += ["--nvtx", "--nvtx-include", "roofline/"]
    else:
        command += ["--launch-skip", str(skip), "--launch-count", "1"]
    return parse_ncu(run(command + [binary] + arguments, log, env))


def write_csv(path, records):
    fields = [key for key, value in records[0].items() if not isinstance(value, (dict, list))]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def provenance(out, env, ncu=None):
    sources = {}
    for name in ["FP16_GEMM_TensorCore", "cs-roofline-toolkit", "nersc-roofline"]:
        directory = ROOT / name
        hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(directory.rglob("*"))
                  if p.is_file() and p.suffix in {".cu", ".cuh", ".cxx", ".c", ".h", ".py"}
                  and ".git" not in p.parts and "__pycache__" not in p.parts}
        sources[name] = {"sha256": hashes}
        if (directory / ".git").exists():
            result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                    capture_output=True, text=True)
            sources[name]["commit"] = result.stdout.strip() if result.returncode == 0 else None
    sources["experiment_scripts"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((ROOT / "scripts").glob("*")) if p.is_file()}
    dump(out / "sources.json", sources)
    if ncu:
        run([ncu, "--version"], out / "raw/ncu_version.log", env)


def hardware(args, out, info, nvcc, env):
    ncu = executable("ncu", args.ncu)
    provenance(out, env, ncu)
    raw = out / "raw/calibration"
    samples = {}
    for mode in ["L1", "L2", "DRAM", "tensor"]:
        records = []
        for trial in range(args.repeats):
            log = raw / f"{mode}_{trial}.log"
            counters = profile(ncu, out / "build/calibrate", [mode], log, env)
            spec = json_line(log.read_text())
            counters.update(spec)
            if spec["validation"] != "PASS":
                raise RuntimeError(f"校准验证失败：{mode}")
            # Only the bandwidth at this interface is used as its roof.
            counters["GBps"] = counters.get(mode + "_bytes", 0) / counters["seconds"] / 1e9
            counters["TFLOPS"] = spec["flops"] / counters["seconds"] / 1e12
            records.append(counters)
        samples[mode] = records
        dump(out / "calibration_samples.json", samples)
    # Run the actual upstream ERT driver, kernel, and postprocessing.
    # New output directory per invocation avoids ERT silently reusing old trials.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config = out / "ert_config.txt"
    ert_result = out / "ert" / stamp
    ert_result.parent.mkdir(parents=True, exist_ok=True)
    sms = info["sms"]
    config.write_text(f"""# Generated for {info['name']}; FP32 CUDA-core ERT, not Tensor Core.
ERT_RESULTS {ert_result}
ERT_DRIVER driver1
ERT_KERNEL kernel1
ERT_GPU True
ERT_GPU_CFLAGS -x cu -arch=sm_{info['sm']}
ERT_GPU_LDFLAGS
ERT_FLOPS 1,2,64,1024
ERT_ALIGN 32
ERT_CC {nvcc}
ERT_CFLAGS -O3 -std=c++17
ERT_LD {nvcc}
ERT_LDFLAGS
ERT_LDLIBS
ERT_RUN ERT_CODE
ERT_PRECISION FP32
ERT_BLOCKS_THREADS {sms * 256},{sms * 4 * 256}
ERT_GPU_BLOCKS {sms},{sms * 4}
ERT_GPU_THREADS 256
ERT_NUM_EXPERIMENTS {args.repeats}
ERT_MEMORY_MAX {args.ert_memory_mib * 1024**2}
ERT_WORKING_SET_MIN 8192
ERT_WSS_MULT 2.0
ERT_TRIALS_MIN 1
ERT_GNUPLOT gnuplot
""")
    run([sys.executable, ERT / "ert", "--no-gnuplot", config], out / "raw/ert.log", env, 1800)
    native = list(ert_result.glob("Run.*/roofline.json"))
    if len(native) != 1:
        raise RuntimeError("ERT 没有生成唯一的 roofline.json")
    ert_json = json.loads(native[0].read_text())
    points = []
    for path in ert_result.rglob("try.[0-9][0-9][0-9]"):
        text = path.read_text()
        if "Last GPU error" in text:
            raise RuntimeError(f"ERT CUDA 错误：{path}")
        for line in text.splitlines():
            match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*", line)
            if match:
                working_set, trials, us, byte_count, flops = map(float, match.groups())
                if us <= 0:
                    raise RuntimeError(f"ERT 时间无效：{path}")
                points.append({"working_set_bytes": int(working_set), "trials": int(trials),
                               "microseconds": us, "logical_bytes": int(byte_count), "flops": int(flops),
                               "GBps": byte_count / us / 1e3, "GFLOPS": flops / us / 1e3,
                               "source": str(path.relative_to(ROOT))})
    if not points:
        raise RuntimeError("ERT 没有有效原始数据")
    write_csv(out / "ert_samples.csv", points)
    ceilings = {mode: max(samples[mode], key=lambda r: r["GBps"])["GBps"]
                for mode in ["L1", "L2", "DRAM"]}
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": info,
              "shared_gpu": info.get("shared_gpu", True),
              "memory_label": "DRAM (GDDR6)" if "A6000" in info["name"] else "DRAM",
              "bandwidth_GBps": ceilings,
              "tensor_fp16_TFLOPS": max(r["TFLOPS"] for r in samples["tensor"]),
              "ert_fp32_TFLOPS": max(value for name, value in ert_json["empirical"]["gflops"]["data"]
                                     if "FP32" in name) / 1000,
              "ert_fp32_max_sample_TFLOPS": max(r["GFLOPS"] for r in points) / 1000,
              "ert_max_logical_GBps": max(r["GBps"] for r in points),
              "ert_native_json": str(native[0].relative_to(ROOT)),
              "ert_native": ert_json, "calibration_samples": samples,
              "roof_policy": "FP32: native ERT summary; bandwidth/Tensor: maximum repeated calibration; not isolated hardware specifications"}
    if "RTX A6000" in info["name"]:
        result["theoretical"] = {
            "fp32_TFLOPS": 38.7, "tensor_fp16_dense_TFLOPS": 154.8,
            "tensor_fp16_sparse_TFLOPS": 309.6, "DRAM_GBps": 768.0,
            "source": "paper reference/nvidia-ampere-whitepaper/nvidia-ampere-whitepaper.pdf",
            "source_table": "Table 3, printed pages 15–16; 1800 MHz GPU boost clock",
            "note": "Dense GEMMs use the dense ceiling; structured 2:4 sparsity is not enabled."
        }
    dump(out / "hardware.json", result)
    report(out)


def operators(args, out, info, nvcc, env):
    ncu = executable("ncu", args.ncu)
    provenance(out, env, ncu)
    shape = [args.m, args.n, args.k]
    # v1-v8 CTA shapes include 128x192 and 192x192; CUTLASS uses 128x256.
    if args.m % 384 or args.n % 768 or args.k % 64:
        raise RuntimeError("统一尺寸要求 M 为 384 倍数，N 为 768 倍数，K 为 64 倍数")
    records = []
    cutlass = Path(args.cutlass_root).expanduser() if args.cutlass_root else ROOT.parent / "cutlass"
    if not (cutlass / "include/cutlass/cutlass.h").is_file():
        raise RuntimeError("找不到 CUTLASS 头文件，请指定 --cutlass-root 或 CUTLASS_ROOT")
    sources = json.loads((out / "sources.json").read_text())
    cutlass_revision = subprocess.run(["git", "-C", str(cutlass), "rev-parse", "HEAD"],
                                      capture_output=True, text=True)
    sources["cutlass_headers"] = {
        "path": str(cutlass.resolve()),
        "commit": cutlass_revision.stdout.strip() if cutlass_revision.returncode == 0 else None,
        "cutlass.h_sha256": hashlib.sha256((cutlass / "include/cutlass/cutlass.h").read_bytes()).hexdigest()
    }
    dump(out / "sources.json", sources)
    dump(out / "operators.json", {"device": info, "shape": shape,
                                  "operators": [], "complete": False})
    (out / "operators.csv").unlink(missing_ok=True)
    for variant in VARIANTS:
        directory = ROOT / "FP16_GEMM_TensorCore" / variant
        binary = out / "build" / variant
        command = [nvcc, "-O3", "-std=c++17", f"-arch=sm_{info['sm']}",
                   "-I", ROOT / "scripts", directory / "FP16_GEMM_TensorCore.cu",
                   ROOT / "scripts/operator_runner.cu", "-o", binary]
        if variant == "cutlass":
            command += ["--expt-relaxed-constexpr", "-I", cutlass / "include"]
        if variant == "cublas":
            command += ["-lcublas"]
        run(command, out / f"raw/operators/{variant}_compile.log", env)
        arguments = [str(v) for v in shape + [args.warmup, args.iterations]]
        timings, counters = [], []
        for trial in range(args.repeats):
            timings.append(json_line(run([binary] + arguments,
                                          out / f"raw/operators/{variant}_timing_{trial}.log", env)))
            counters.append(profile(ncu, binary, arguments,
                                    out / f"raw/operators/{variant}_ncu_{trial}.log", env, skip=None))
        # Keep time and traffic paired within each profile, choose the median
        # time sample rather than mixing independently selected maxima.
        selected = sorted(counters, key=lambda r: r["seconds"])[len(counters) // 2]
        flops = 2 * math.prod(shape)
        record = {"operator": variant, "M": args.m, "N": args.n, "K": args.k,
                  "flops": flops, "profile_ms": selected["seconds"] * 1000,
                  "TFLOPS": flops / selected["seconds"] / 1e12,
                  "event_ms": statistics.median(r["event_ms"] for r in timings),
                  "event_TFLOPS": statistics.median(r["TFLOPS"] for r in timings),
                  "validation": "PASS", "profile_samples": counters, "timing_samples": timings}
        for level in ["L1", "L2", "DRAM"]:
            traffic = selected[level + "_bytes"]
            if traffic <= 0:
                raise RuntimeError(f"{variant} 的 {level} 流量为零，无法计算实测 AI")
            record[level + "_bytes"] = traffic
            record["AI_" + level] = flops / traffic
        records.append(record)
        dump(out / "operators.json", {"device": info, "shape": shape, "operators": records,
                                      "complete": len(records) == len(VARIANTS)})
        write_csv(out / "operators.csv", records)
        print(f"{variant}: {record['TFLOPS']:.3f} TFLOP/s; correctness PASS", flush=True)
    report(out)


def report(out):
    hardware_path, operators_path = out / "hardware.json", out / "operators.json"
    lines = [f"# {out.name} 分层 Roofline 实测结果", ""]
    if hardware_path.exists():
        h = json.loads(hardware_path.read_text())
        d = h["device"]
        lines += [f"设备：{d['name']}；SM {d['sm']}；{d['sms']} 个 SM；显存 {d['memory_bytes'] / 2**30:.1f} GiB；L2 {d['l2_bytes'] / 2**20:.1f} MiB。",
                  f"测量时间（UTC）：{h['timestamp_utc']}。", "",
                  ("本次运行时 GPU 上有其他计算负载。以下为共享环境实测最大值，不能作为独占设备的标称峰值。"
                   if h.get("shared_gpu") else "采集开始时未发现其他 GPU 计算进程；以下为本次环境实测最大值。"),
                  "A6000 的显存是 GDDR6；本文 DRAM 对应任务中所称的 HBM 层。" if "A6000" in d["name"] else "DRAM 表示设备显存层。", "",
                  "| 实测指标 | 数值 | 方法 |", "|---|---:|---|",
                  f"| FP32 CUDA core | {h['ert_fp32_TFLOPS']:.3f} TFLOP/s | 原生 ERT roofline.json 汇总 |",
                  f"| Dense FP16 Tensor Core | {h['tensor_fp16_TFLOPS']:.3f} TFLOP/s | 8 个独立 WMMA 累加器，FP16 累加 |"]
        for level, bandwidth in h["bandwidth_GBps"].items():
            lines.append(f"| {level} | {bandwidth:.3f} GB/s | 对应接口实测计数器字节数 / kernel 时间 |")
        if "ert_fp32_max_sample_TFLOPS" in h:
            lines += ["", f"ERT 全部原始单次样本的最高值为 {h['ert_fp32_max_sample_TFLOPS']:.3f} TFLOP/s；上表和图使用原生 ERT 后处理汇总值（重复实验取中位时间，再选平台峰值）。"]
        if "theoretical" in h:
            t = h["theoretical"]
            lines += ["", "白皮书 Table 3（印刷页 15–16，1800 MHz Boost）理论值：",
                      f"FP32 **{t['fp32_TFLOPS']} TFLOP/s**；稠密 FP16 Tensor Core **{t['tensor_fp16_dense_TFLOPS']} TFLOP/s**；GDDR6 **{t['DRAM_GBps']} GB/s**。",
                      f"2:4 稀疏 Tensor Core 为 {t['tensor_fp16_sparse_TFLOPS']} TFLOP/s，本实验为稠密 GEMM，因此使用稠密上限。图中虚线表示理论值，实线表示本次实测。",
                      "理论值基于标称 Boost，实测未锁频，实际 Boost 较高时实测可略高于标称理论值。"]
        lines += ["", "带宽和 Tensor Core 上限取重复校准的最大值。ERT 标量 FP32 上限仅作参考；FP16 GEMM 图使用单独实测的 FP16 Tensor Core 上限。",
                  "ERT 的 read/write logical bandwidth 不能直接当作 L1/L2/DRAM 的物理接口带宽，故另用 `.ca`/`.cg` 加载探针配合计数器校准。",
                  "L1 探针使用每 CTA 4 KiB 重复 `.ca` 读取；L2 使用一半 L2 容量的 `.cg` 工作集；DRAM 使用至少 16 倍 L2、且至少 64 MiB 的 `.cg` 工作集。",
                  "L1 指标定义为 L1TEX 接口的请求 sector（包含绕过 L1 缓存的访问），不包括 shared-memory 的内部数据流量。", "",
                  f"ERT 原生输出：[{h['ert_native_json']}](../{h['ert_native_json']})。", ""]
    if operators_path.exists():
        o = json.loads(operators_path.read_text())
        lines += [f"统一矩阵尺寸 M×N×K = {'×'.join(map(str, o['shape']))}；C=A×B，FP16 输入与输出；执行量按有效算法 FLOP `2*M*N*K` 计算，FMA=2 FLOP。", "",
                  "所有算子在采样的 128 个输出位置上通过 CPU FP64 参考验证；这不等同于全矩阵逐元素验证。",
                  "每层 AI = 有效 FLOP / 实测接口字节数。L1/L2 sector 大小均为 32 B。",
                  "主图性能与 AI 使用同一次 ncu 样本；取重复样本中 kernel 总时间的中位样本。Event 列来自另一次无 profiler 的 CUDA Event 测量。",
                  "NVTX 范围只包围一次预热后的完整 GEMM 调用；包含该调用的所有辅助 kernel，排除初始化、验证和预热。ncu 每次 replay 清缓存，普通计时允许预热缓存，因此两者性能可能不同。", "",
                  "| 算子 | ncu TFLOP/s | Event TFLOP/s | AI L1 (FLOP/B) | AI L2 | AI DRAM | 验证 |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for r in o["operators"]:
            lines.append(f"| {r['operator']} | {r['TFLOPS']:.3f} | {r['event_TFLOPS']:.3f} | {r['AI_L1']:.3f} | {r['AI_L2']:.3f} | {r['AI_DRAM']:.3f} | {r['validation']} |")
        if o.get("complete"):
            best = max(o["operators"], key=lambda r: r["TFLOPS"])
            baseline = next(r for r in o["operators"] if r["operator"] == "v1")
            lines += ["", f"在本次统一尺寸与采集条件下，最快的实现为 {best['operator']}，ncu 实测 {best['TFLOPS']:.3f} TFLOP/s，约为 v1 的 {best['TFLOPS'] / baseline['TFLOPS']:.2f} 倍。"]
        if not o.get("complete"):
            lines += ["", "尚未完成全部 10 个算子的测量。"]
        lines += ["", "详细字节数、时间和每次采样见 `operators.csv` / `operators.json`；原始编译和 profiler 日志见 `raw/`。", ""]
    if (out / "roofline.png").exists():
        lines += [f"![分层 Roofline]({out.name}/roofline.png)", "",
                  f"[SVG 矢量图]({out.name}/roofline.svg) · [PDF]({out.name}/roofline.pdf) · [ERT 工作集曲线]({out.name}/ert_working_set.png)", ""]
    lines += ["结果中的上限和算子可能受不同时间的共享负载、时钟及测量重放影响，超出实测屋顶不应解释为超越硬件理论峰值。", ""]
    (out.parent / f"{out.name}.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "ert", "operators", "plot"])
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--nvcc", default=os.environ.get("NVCC"))
    parser.add_argument("--ncu", default=os.environ.get("NCU"))
    parser.add_argument("--cutlass-root", default=os.environ.get("CUTLASS_ROOT"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--m", type=int, default=3072)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ert-memory-mib", type=int, default=64)
    args = parser.parse_args()
    if any(v <= 0 for v in [args.m, args.n, args.k, args.warmup, args.iterations, args.repeats]):
        parser.error("尺寸、预热次数、迭代次数和重复次数必须为正数")
    if args.ert_memory_mib < 32:
        parser.error("--ert-memory-mib 必须至少为 32")
    if "," in args.device:
        parser.error("本实验使用单张 GPU；--device 请指定一个 GPU ID 或 UUID")
    if args.stage == "plot":
        if args.output is None:
            parser.error("离线画图需指定 --output resluts/A6000")
        out = args.output.resolve()
    else:
        nvcc = executable("nvcc", args.nvcc)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.device
        # Use our own TMPDIR: shared /tmp lock files can belong to other users.
        temp = ROOT / ".cache/ncu-tmp"
        temp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(temp)
        env["PATH"] = str(Path(nvcc).parent) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        # Bootstrap device discovery independently of any Python CUDA package.
        bootstrap = ROOT / ".cache/calibrate"
        run([nvcc, "-O3", "-std=c++17", "-arch=native", ROOT / "scripts/calibrate.cu", "-o", bootstrap],
            ROOT / ".cache/bootstrap_compile.log", env)
        info = json_line(run([bootstrap, "info"], ROOT / ".cache/device.log", env))
        name = info["name"]
        model = re.search(r"\b(?:[AH]\d{3,4}|V100|T4|L4|L40S?)\b", name)
        label = model.group() if model else re.sub(r"[^\w-]+", "_", name).strip("_")
        out = args.output.resolve() if args.output else ROOT / "resluts" / label
        (out / "build").mkdir(parents=True, exist_ok=True)
        shutil.copy2(bootstrap, out / "build/calibrate")
        snapshot = run(["nvidia-smi", "--id", args.device], out / f"raw/{args.stage}_environment.log", env)
        info["shared_gpu"] = bool(re.search(r"\|\s+\d+\s+\S+\s+\S+\s+\d+\s+[CG](?:\+G)?\s+", snapshot))
        dump(out / "device.json", info)
        run([nvcc, "--version"], out / "raw/nvcc_version.log", env)
        if args.stage in ["all", "ert"]:
            hardware(args, out, info, nvcc, env)
        if args.stage in ["all", "operators"]:
            operators(args, out, info, nvcc, env)
    if args.stage in ["all", "plot"]:
        subprocess.run([sys.executable, str(ROOT / "nersc-roofline/Plotting/plot_roofline_hierarchical.py"),
                        "--hardware", str(out / "hardware.json"), "--operators", str(out / "operators.json"),
                        "--output", str(out / "roofline")], check=True)
        report(out)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
