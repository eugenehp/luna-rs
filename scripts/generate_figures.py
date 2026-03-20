#!/usr/bin/env python3
"""Generate unified benchmark comparison charts for README."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────

root = Path(__file__).resolve().parent.parent
linux_dir = root / "bench_results" / "20260320_125822_-"
mac_dir = root / "bench_results" / "20260320_172301_apple_m3_max"
out_dir = root / "figures"
out_dir.mkdir(exist_ok=True)

linux_cpu = json.loads((linux_dir / "cpu.json").read_text())
linux_gpu = json.loads((linux_dir / "gpu.json").read_text())
mac_cpu = json.loads((mac_dir / "cpu.json").read_text())
mac_gpu = json.loads((mac_dir / "gpu.json").read_text())

variants = [d["variant"].capitalize() for d in linux_cpu]

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.6,
    "font.size": 11,
    "font.family": "sans-serif",
})

LINUX_CPU_COLOR = "#4ecdc4"
LINUX_GPU_COLOR = "#ff6b6b"
MAC_CPU_COLOR = "#ffd93d"
MAC_GPU_COLOR = "#c084fc"

# ── Chart 1: Inference Latency (all 4 backends) ─────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(variants))
width = 0.2

datasets = [
    (linux_cpu, "Linux CPU (aarch64 16C)", LINUX_CPU_COLOR),
    (linux_gpu, "Linux GPU (Virtio/Vulkan)", LINUX_GPU_COLOR),
    (mac_cpu, "M3 Max CPU (Accelerate)", MAC_CPU_COLOR),
    (mac_gpu, "M3 Max GPU (Metal)", MAC_GPU_COLOR),
]

for i, (data, label, color) in enumerate(datasets):
    means = [d["inference"]["mean_ms"] for d in data]
    stds = [d["inference"]["std_ms"] for d in data]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, means, width, yerr=stds, label=label, color=color,
                  alpha=0.85, edgecolor="white", linewidth=0.5, capsize=3)
    for bar in bars:
        h = bar.get_height()
        if h < 100:
            ax.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h + 15, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xlabel("LUNA Variant")
ax.set_ylabel("Inference Latency (ms)")
ax.set_title("LUNA-RS Inference Latency — All Backends\n(22 channels × 1280 samples, 5s @ 256Hz)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"LUNA-{v}" for v in variants])
ax.legend(loc="upper left", framealpha=0.8, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.set_yscale("log")
ax.set_ylabel("Inference Latency (ms, log scale)")
plt.tight_layout()
fig.savefig(out_dir / "inference_latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ {out_dir}/inference_latency.png")

# ── Chart 2: Load Time ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
for i, (data, label, color) in enumerate(datasets):
    loads = [d["load_ms"] for d in data]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, loads, width, label=label, color=color,
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 20, f"{h:.0f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xlabel("LUNA Variant")
ax.set_ylabel("Load Time (ms)")
ax.set_title("Model Load Time — All Backends", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"LUNA-{v}" for v in variants])
ax.legend(loc="upper left", framealpha=0.8, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(out_dir / "load_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ {out_dir}/load_time.png")

# ── Chart 3: Latency Distribution (box plot) ────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
positions = []
labels = []
all_data = []
colors_list = []
pos = 1

backend_labels = [
    ("Linux\nCPU", LINUX_CPU_COLOR),
    ("Linux\nGPU", LINUX_GPU_COLOR),
    ("M3 Max\nCPU", MAC_CPU_COLOR),
    ("M3 Max\nGPU", MAC_GPU_COLOR),
]
all_datasets = [linux_cpu, linux_gpu, mac_cpu, mac_gpu]

for vi, variant in enumerate(variants):
    for di, (data, (blabel, bcolor)) in enumerate(zip(all_datasets, backend_labels)):
        all_data.append(data[vi]["inference"]["all_ms"])
        positions.append(pos)
        labels.append(f"{variant}\n{blabel}")
        colors_list.append(bcolor)
        pos += 1
    pos += 1  # gap between variants

bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                showmeans=True, meanline=True,
                meanprops=dict(color="white", linewidth=1.5),
                medianprops=dict(color="#1a1a2e", linewidth=1.5),
                whiskerprops=dict(color="#e0e0e0"),
                capprops=dict(color="#e0e0e0"),
                flierprops=dict(markerfacecolor="#e0e0e0", markersize=4))
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Latency (ms)")
ax.set_title("Inference Latency Distribution (22ch × 1280 samples, 10 runs)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.set_yscale("log")
ax.set_ylabel("Latency (ms, log scale)")
plt.tight_layout()
fig.savefig(out_dir / "latency_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ {out_dir}/latency_distribution.png")

# ── Chart 4: Channel Scaling ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

for vi, variant in enumerate(variants):
    ax = axes[0][vi]
    for data, label, color, marker in [
        (linux_cpu, "Linux CPU", LINUX_CPU_COLOR, "o"),
        (linux_gpu, "Linux GPU", LINUX_GPU_COLOR, "s"),
        (mac_cpu, "M3 Max CPU", MAC_CPU_COLOR, "^"),
        (mac_gpu, "M3 Max GPU", MAC_GPU_COLOR, "D"),
    ]:
        cs = data[vi]["channel_scaling"]
        chans = [c["channels"] for c in cs]
        means = [c["mean_ms"] for c in cs]
        ax.plot(chans, means, f"{marker}-", color=color, linewidth=2, markersize=6,
                label=label, zorder=3)
    ax.set_xlabel("Channels")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"LUNA-{variant}", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.8, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xticks(chans)
    ax.set_yscale("log")

fig.suptitle("Channel Scaling — Latency vs Channel Count (T=1280)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(out_dir / "channel_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ {out_dir}/channel_scaling.png")

# ── Chart 5: Speedup summary ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(variants))
width = 0.25

# Speedup relative to Linux CPU (baseline)
linux_cpu_means = [d["inference"]["mean_ms"] for d in linux_cpu]
comparisons = [
    (linux_gpu, "Linux GPU (Virtio/Vulkan)", LINUX_GPU_COLOR),
    (mac_cpu, "M3 Max CPU (Accelerate)", MAC_CPU_COLOR),
    (mac_gpu, "M3 Max GPU (Metal)", MAC_GPU_COLOR),
]

for i, (data, label, color) in enumerate(comparisons):
    speedups = [lc / d["inference"]["mean_ms"] for lc, d in zip(linux_cpu_means, data)]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, speedups, width, label=label, color=color,
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(y=1.0, color=LINUX_CPU_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
           label="Linux CPU (baseline)")
ax.set_xlabel("LUNA Variant")
ax.set_ylabel("Speedup vs Linux CPU")
ax.set_title("Speedup Relative to Linux CPU Baseline",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"LUNA-{v}" for v in variants])
ax.legend(loc="upper left", framealpha=0.8, fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(out_dir / "speedup.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ {out_dir}/speedup.png")

print("\nDone — all figures saved to ./figures/")
