#!/usr/bin/env python3
"""Python 3 NERSC hierarchical Roofline plot: legacy text or measured JSON."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = {"L1": "#007c91", "L2": "#dc8b15", "DRAM": "#a448a2"}
MARKERS = ["o", "s", "^", "v", "D", ">", "<", "P", "X", "*"]


def read_legacy(path):
    data = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, values = line.partition(" ")
        data[key] = re.findall(r"'([^']*)'", values) if "'" in values else [float(v) for v in values.split()]
    roofs = dict(zip(data["mem_roof_names"], data["memroofs"]))
    roofs = {"DRAM" if key == "HBM" else key: value for key, value in roofs.items()}
    records = []
    for i, performance in enumerate(data.get("FLOPS", [])):
        records.append({"operator": data["labels"][i], "TFLOPS": performance / 1000,
                        "AI_L1": data["AI_L1"][i], "AI_L2": data["AI_L2"][i],
                        "AI_DRAM": data["AI_HBM"][i]})
    hardware = {"device": {"name": "Hierarchical Roofline"}, "bandwidth_GBps": roofs,
                "tensor_fp16_TFLOPS": max(data["comproofs"]) / 1000,
                "legacy_compute_roofs": list(zip(data["comp_roof_names"], data["comproofs"])),
                "shared_gpu": False, "memory_label": "DRAM / HBM"}
    return hardware, records


def plot(hardware, records, output):
    roofs = hardware["bandwidth_GBps"]
    peak = hardware["tensor_fp16_TFLOPS"] * 1000
    theory = hardware.get("theoretical", {})
    theoretical_peak = theory.get("tensor_fp16_dense_TFLOPS", 0) * 1000
    fp32_peak = hardware.get("ert_fp32_TFLOPS", 0) * 1000
    levels = [level for level in ["L1", "L2", "DRAM"] if level in roofs]
    intensities = [r["AI_" + level] for r in records for level in levels] or [peak / b for b in roofs.values()]
    if any(value <= 0 or not np.isfinite(value) for value in intensities + [peak] + list(roofs.values())):
        raise ValueError("Roofline data must be finite and positive")
    xmin = min(min(intensities) / 3, min(peak / b for b in roofs.values()) / 8)
    xmax = max(max(intensities) * 2, max(peak / b for b in roofs.values()) * 2)
    x = np.geomspace(xmin, xmax, 1000)
    lowest = min((r["TFLOPS"] * 1000 for r in records), default=peak)
    ymin = min(lowest / 3, min(roofs.values()) * xmin / 2)
    ymax = max(peak, theoretical_peak, max((r["TFLOPS"] * 1000 for r in records), default=peak)) * 2.5

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(14, 9), layout="constrained")
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin, ymax),
           xlabel="Arithmetic intensity [effective FLOP / measured byte]",
           ylabel="Performance [GFLOP/s]")
    for level in levels:
        bandwidth = roofs[level]
        name = hardware.get("memory_label", "DRAM") if level == "DRAM" else level
        ax.plot(x, np.minimum(bandwidth * x, peak), color=COLORS[level], lw=2.2,
                label=f"{name}: {bandwidth:,.1f} GB/s")
    ax.axhline(peak, color="#374151", linestyle="-", lw=1.5,
               label=f"Measured dense FP16 Tensor: {peak / 1000:.2f} TFLOP/s" if "legacy_compute_roofs" not in hardware
               else f"Compute: {peak:,.1f} GFLOP/s")
    if fp32_peak:
        ax.axhline(fp32_peak, color="#2563eb", linestyle="-", lw=1.5,
                   label=f"Measured FP32 (ERT): {fp32_peak / 1000:.2f} TFLOP/s")
    if theory:
        ax.axhline(theory["fp32_TFLOPS"] * 1000, color="#2563eb", ls="--", lw=1.5,
                   label=f"Theoretical FP32: {theory['fp32_TFLOPS']:.1f} TFLOP/s")
        ax.axhline(theoretical_peak, color="#374151", ls="--", lw=1.5,
                   label=f"Theoretical dense FP16 Tensor: {theoretical_peak / 1000:.1f} TFLOP/s")
        ax.plot(x, np.minimum(theory["DRAM_GBps"] * x, theoretical_peak),
                color=COLORS["DRAM"], ls="--", lw=1.8,
                label=f"Theoretical GDDR6: {theory['DRAM_GBps']:.0f} GB/s")
    for name, value in hardware.get("legacy_compute_roofs", []):
        if value < peak:
            ax.axhline(value, color="gray", linestyle=":", label=f"{name}: {value:,.1f} GFLOP/s")
    operator_handles = []
    for i, record in enumerate(records):
        marker = MARKERS[i % len(MARKERS)]
        ais = [record["AI_" + level] for level in levels]
        gflops = record["TFLOPS"] * 1000
        ax.plot(ais, [gflops] * len(ais), color="#aab1bc", lw=0.8, alpha=0.7, zorder=2)
        for level, ai in zip(levels, ais):
            ax.scatter(ai, gflops, marker=marker, color=COLORS[level], s=90,
                       edgecolors="white", linewidths=0.5, zorder=3)
        operator_handles.append(Line2D([], [], color="#4b5563", marker=marker,
                                       linestyle="None", markersize=8, label=record["operator"]))
    roof_legend = ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    ax.add_artist(roof_legend)
    if operator_handles:
        ax.legend(handles=operator_handles, loc="lower right", ncols=2, title="GEMM implementation", fontsize=10)
    ax.grid(which="major", alpha=0.2)
    title = hardware["device"]["name"] + " — Hierarchical Roofline"
    if records and records[0].get("M"):
        r = records[0]
        title += f"\nFP16 GEMM: {r['M']} × {r['N']} × {r['K']}"
    ax.set_title(title, pad=15)
    note = "Points use matched kernel duration and interface traffic. Lines connect the same implementation."
    if theory:
        note += "\nSolid: measured. Dashed: whitepaper Table 3 (1800 MHz). Dense GEMMs; no 2:4 sparsity."
    if hardware.get("shared_gpu"):
        note += "\nShared GPU workload: empirical roofs are observed values, not isolated hardware peaks."
    fig.supxlabel(note, fontsize=9, color="#4b5563")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)

    # Do not relabel ERT logical read/write traffic as physical cache bandwidth.
    samples = output.parent / "ert_samples.csv"
    if samples.exists():
        with samples.open() as f:
            rows = list(csv.DictReader(f))
        working_sets = sorted({int(r["working_set_bytes"]) for r in rows})
        best_bw = [max(float(r["GBps"]) for r in rows if int(r["working_set_bytes"]) == size) for size in working_sets]
        best_flops = [max(float(r["GFLOPS"]) for r in rows if int(r["working_set_bytes"]) == size) for size in working_sets]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for ax, values, label in zip(axes, [best_bw, best_flops], ["Logical bandwidth [GB/s]", "FP32 performance [GFLOP/s]"]):
            ax.loglog(working_sets, values, "o-", color="#007c91")
            ax.set(xlabel="Working set [bytes]", ylabel=label)
            ax.grid(which="major", alpha=0.2)
        fig.suptitle(hardware["device"]["name"] + " — original ERT measurements (best per working set)")
        fig.savefig(output.parent / "ert_working_set.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, help="original NERSC text format")
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--operators", type=Path)
    parser.add_argument("--output", type=Path, default=Path("roofline_hierarchical"))
    args = parser.parse_args()
    if args.data:
        hardware, records = read_legacy(args.data)
    elif args.hardware and args.operators:
        hardware = json.loads(args.hardware.read_text())
        data = json.loads(args.operators.read_text())
        if not data.get("complete"):
            parser.error("全部 10 个算子采集完成后才能生成最终图片")
        if hardware["device"]["name"] != data["device"]["name"]:
            parser.error("硬件与算子测量来自不同的 GPU 型号")
        records = data["operators"]
    else:
        parser.error("provide a legacy data file or both --hardware and --operators")
    plot(hardware, records, args.output)
    print(f"Saved {args.output}.png / .svg / .pdf")


if __name__ == "__main__":
    main()
