#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# bench.sh — LUNA-RS CPU vs GPU inference benchmark
#
# Detects platform (macOS/Linux), CPU model, GPU, runs benchmarks for both
# backends, saves JSON results, and generates comparison charts.
#
# Usage:
#   ./bench.sh                      # benchmark base model
#   ./bench.sh base,large,huge      # benchmark all variants
#   ./bench.sh base 10 20           # 10 warmup, 20 timed runs
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VARIANTS="${1:-base,large,huge}"
WARMUP="${2:-3}"
RUNS="${3:-10}"

# ─── Platform Detection ─────────────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"

detect_cpu() {
    local result=""
    case "$OS" in
        Darwin)
            result="$(sysctl -n machdep.cpu.brand_string 2>/dev/null \
                || sysctl -n hw.model 2>/dev/null \
                || echo "")"
            ;;
        Linux)
            result="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null \
                | sed 's/model name\s*:\s*//' || echo "")"
            if [ -z "$result" ]; then
                result="$(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/.*Model name:\s*//' || echo "")"
            fi
            if [ -z "$result" ]; then
                # Fallback: architecture + nproc
                result="${ARCH} $(nproc 2>/dev/null || echo '?')cores"
            fi
            ;;
    esac
    echo "${result:-Unknown_CPU}"
}

detect_gpu() {
    case "$OS" in
        Darwin)
            # Apple Silicon — GPU is the chip itself
            local chip
            chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '')"
            if [[ "$chip" == *"Apple"* ]]; then
                echo "$chip GPU"
            else
                system_profiler SPDisplaysDataType 2>/dev/null \
                    | grep -m1 'Chipset Model' \
                    | sed 's/.*Chipset Model: //' \
                    || echo "Unknown"
            fi
            ;;
        Linux)
            # Try nvidia-smi first
            if command -v nvidia-smi &>/dev/null; then
                nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                    | head -1 \
                    || echo "Unknown NVIDIA"
            # Try lspci for AMD/Intel
            elif command -v lspci &>/dev/null; then
                lspci 2>/dev/null \
                    | grep -i 'vga\|3d\|display' \
                    | head -1 \
                    | sed 's/.*: //' \
                    || echo "Unknown"
            else
                echo "Unknown"
            fi
            ;;
        *) echo "Unknown" ;;
    esac
}

detect_cpu_features() {
    case "$OS" in
        Darwin)
            local cores threads
            cores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo '?')"
            threads="$(sysctl -n hw.logicalcpu 2>/dev/null || echo '?')"
            local mem
            mem="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
            echo "${cores}C/${threads}T, ${mem}GB RAM"
            ;;
        Linux)
            local cores threads
            cores="$(nproc --all 2>/dev/null || echo '?')"
            threads="$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo '?')"
            local mem
            mem="$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"
            echo "${cores}C/${threads}T, ${mem}GB RAM"
            ;;
        *) echo "unknown" ;;
    esac
}

detect_gpu_backend() {
    case "$OS" in
        Darwin) echo "metal" ;;
        Linux)
            if command -v nvidia-smi &>/dev/null || command -v vulkaninfo &>/dev/null; then
                echo "vulkan"
            else
                echo "wgpu"
            fi
            ;;
        *) echo "wgpu" ;;
    esac
}

detect_cpu_backend_features() {
    case "$OS" in
        Darwin) echo "ndarray,blas-accelerate,hf-download" ;;
        *)
            if ldconfig -p 2>/dev/null | grep -q libopenblas; then
                echo "ndarray,openblas-system,hf-download"
            else
                echo "ndarray,hf-download"
            fi
            ;;
    esac
}

# ─── Slug: sanitize string for filenames ─────────────────────────────────────

slugify() {
    echo "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9._-]/_/g' \
        | sed 's/__*/_/g' \
        | sed 's/^_//;s/_$//'
}

# ─── Gather info ─────────────────────────────────────────────────────────────

CPU_NAME="$(detect_cpu)"
GPU_NAME="$(detect_gpu)"
CPU_INFO="$(detect_cpu_features)"
GPU_BACKEND="$(detect_gpu_backend)"
CPU_FEATURES="$(detect_cpu_backend_features)"

CPU_SLUG="$(slugify "$CPU_NAME")"
GPU_SLUG="$(slugify "$GPU_NAME")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATE_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"

RESULTS_DIR="bench_results"
mkdir -p "$RESULTS_DIR"

RUN_ID="${TIMESTAMP}_${CPU_SLUG}"
RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LUNA-RS — CPU vs GPU Benchmark                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Platform:   $OS $ARCH"
echo "  CPU:        $CPU_NAME ($CPU_INFO)"
echo "  GPU:        $GPU_NAME"
echo "  Variants:   $VARIANTS"
echo "  Warmup:     $WARMUP    Runs: $RUNS"
echo "  Output:     $RUN_DIR/"
echo ""

# ─── Write metadata ──────────────────────────────────────────────────────────

cat > "$RUN_DIR/meta.json" <<EOF
{
  "timestamp": "$DATE_HUMAN",
  "os": "$OS",
  "arch": "$ARCH",
  "cpu": "$CPU_NAME",
  "cpu_info": "$CPU_INFO",
  "gpu": "$GPU_NAME",
  "gpu_backend": "$GPU_BACKEND",
  "variants": "$VARIANTS",
  "warmup": $WARMUP,
  "runs": $RUNS,
  "run_id": "$RUN_ID"
}
EOF

# ─── Build ────────────────────────────────────────────────────────────────────

echo "━━━ Building CPU backend (features: $CPU_FEATURES) ━━━"
cargo build --release --example benchmark --features "$CPU_FEATURES" 2>&1 | tail -3
echo ""

GPU_AVAILABLE=false
GPU_FEATURE="${GPU_BACKEND},hf-download"
echo "━━━ Building GPU backend (features: $GPU_FEATURE) ━━━"
if cargo build --release --example benchmark --no-default-features --features "$GPU_FEATURE" 2>&1 | tail -3; then
    GPU_AVAILABLE=true
    echo ""
else
    echo "  ⚠  GPU build failed — skipping GPU benchmark"
    echo ""
fi

# ─── Per-variant timeout (seconds) ───────────────────────────────────────────
# Generous: 5 minutes per variant should be enough for even LUNA-Huge on slow HW
VARIANT_COUNT=$(echo "$VARIANTS" | tr ',' '\n' | wc -l)
TIMEOUT=$(( VARIANT_COUNT * 300 ))

# ─── Run CPU Benchmark ───────────────────────────────────────────────────────

CPU_JSON="$RUN_DIR/cpu.json"
CPU_LOG="$RUN_DIR/cpu.log"
echo "━━━ Running CPU benchmark (timeout ${TIMEOUT}s) ━━━"
if timeout "$TIMEOUT" cargo run --release --example benchmark --features "$CPU_FEATURES" \
    -- --variants "$VARIANTS" --warmup "$WARMUP" --runs "$RUNS" --json \
    > "$CPU_JSON" 2>"$CPU_LOG"; then
    echo "  ✓ CPU results → $CPU_JSON"
else
    echo "  ⚠  CPU benchmark failed or timed out (see $CPU_LOG)"
fi
echo ""

# ─── Run GPU Benchmark ───────────────────────────────────────────────────────

GPU_JSON="$RUN_DIR/gpu.json"
GPU_LOG="$RUN_DIR/gpu.log"
if $GPU_AVAILABLE; then
    echo "━━━ Running GPU benchmark (timeout ${TIMEOUT}s) ━━━"
    if timeout "$TIMEOUT" cargo run --release --example benchmark --no-default-features --features "$GPU_FEATURE" \
        -- --variants "$VARIANTS" --warmup "$WARMUP" --runs "$RUNS" --json \
        > "$GPU_JSON" 2>"$GPU_LOG"; then
        echo "  ✓ GPU results → $GPU_JSON"
    else
        echo "  ⚠  GPU benchmark failed or timed out (see $GPU_LOG)"
        GPU_AVAILABLE=false
        rm -f "$GPU_JSON"
    fi
    echo ""
fi

# ─── Generate Charts ─────────────────────────────────────────────────────────

echo "━━━ Generating charts ━━━"

python3 - "$RUN_DIR" "$GPU_AVAILABLE" "$CPU_NAME" "$GPU_NAME" "$DATE_HUMAN" <<'PYEOF'
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

run_dir = Path(sys.argv[1])
gpu_available = sys.argv[2] == "true"
cpu_name = sys.argv[3]
gpu_name = sys.argv[4]
date_str = sys.argv[5]

# ── Load data ────────────────────────────────────────────────────────────────

with open(run_dir / "cpu.json") as f:
    cpu_data = json.load(f)

gpu_data = None
if gpu_available and (run_dir / "gpu.json").exists():
    try:
        with open(run_dir / "gpu.json") as f:
            gpu_data = json.load(f)
    except json.JSONDecodeError:
        gpu_available = False

variants = [d["variant"] for d in cpu_data]

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.6,
    "font.size":        11,
    "font.family":      "sans-serif",
})

CPU_COLOR = "#4ecdc4"
GPU_COLOR = "#ff6b6b"
COLORS = ["#4ecdc4", "#ff6b6b", "#ffe66d", "#a8e6cf", "#dda0dd"]

# ── Chart 1: Inference Latency by Variant (CPU vs GPU bar chart) ─────────

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(variants))
width = 0.35

cpu_means = [d["inference"]["mean_ms"] for d in cpu_data]
cpu_stds  = [d["inference"]["std_ms"] for d in cpu_data]

bars_cpu = ax.bar(x - width/2, cpu_means, width, yerr=cpu_stds,
                  label=f"CPU — {cpu_name}", color=CPU_COLOR, alpha=0.85,
                  edgecolor="white", linewidth=0.5, capsize=4)

if gpu_data:
    gpu_means = [d["inference"]["mean_ms"] for d in gpu_data]
    gpu_stds  = [d["inference"]["std_ms"] for d in gpu_data]
    bars_gpu = ax.bar(x + width/2, gpu_means, width, yerr=gpu_stds,
                      label=f"GPU — {gpu_name}", color=GPU_COLOR, alpha=0.85,
                      edgecolor="white", linewidth=0.5, capsize=4)

# Add value labels on bars
for bar in bars_cpu:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
if gpu_data:
    for bar in bars_gpu:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("LUNA Variant")
ax.set_ylabel("Inference Latency (ms)")
ax.set_title("LUNA-RS Inference Latency — CPU vs GPU\n(22 channels × 1280 samples, 5s @ 256Hz)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"LUNA-{v.capitalize()}" for v in variants])
ax.legend(loc="upper left", framealpha=0.8)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "inference_latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/inference_latency.png")

# ── Chart 2: Channel Scaling (one subplot per variant) ───────────────────

n_variants = len(variants)
fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5), squeeze=False)

for i, variant in enumerate(variants):
    ax = axes[0][i]
    cpu_cs = cpu_data[i]["channel_scaling"]
    chans = [c["channels"] for c in cpu_cs]
    cpu_ms = [c["mean_ms"] for c in cpu_cs]

    ax.plot(chans, cpu_ms, "o-", color=CPU_COLOR, linewidth=2, markersize=6,
            label=f"CPU", zorder=3)

    if gpu_data:
        gpu_cs = gpu_data[i]["channel_scaling"]
        gpu_ms = [c["mean_ms"] for c in gpu_cs]
        ax.plot(chans, gpu_ms, "s-", color=GPU_COLOR, linewidth=2, markersize=6,
                label=f"GPU", zorder=3)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"LUNA-{variant.capitalize()}", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xticks(chans)

fig.suptitle("Channel Scaling — Latency vs Channel Count (T=1280)", fontsize=13, fontweight="bold", y=1.02)
fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "channel_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/channel_scaling.png")

# ── Chart 3: Load Times ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(variants))
width = 0.35

cpu_load = [d["load_ms"] for d in cpu_data]
bars = ax.bar(x - width/2, cpu_load, width, label="CPU", color=CPU_COLOR, alpha=0.85,
              edgecolor="white", linewidth=0.5)

if gpu_data:
    gpu_load = [d["load_ms"] for d in gpu_data]
    bars_g = ax.bar(x + width/2, gpu_load, width, label="GPU", color=GPU_COLOR, alpha=0.85,
                    edgecolor="white", linewidth=0.5)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
if gpu_data:
    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("LUNA Variant")
ax.set_ylabel("Load Time (ms)")
ax.set_title("Model Load Time — CPU vs GPU", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"LUNA-{v.capitalize()}" for v in variants])
ax.legend(framealpha=0.8)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "load_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/load_time.png")

# ── Chart 4: Run distribution (box plot) ─────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

positions = []
labels = []
all_data = []
colors = []
pos = 1

for i, variant in enumerate(variants):
    cpu_runs = cpu_data[i]["inference"]["all_ms"]
    all_data.append(cpu_runs)
    positions.append(pos)
    labels.append(f"{variant.capitalize()}\nCPU")
    colors.append(CPU_COLOR)
    pos += 1

    if gpu_data:
        gpu_runs = gpu_data[i]["inference"]["all_ms"]
        all_data.append(gpu_runs)
        positions.append(pos)
        labels.append(f"{variant.capitalize()}\nGPU")
        colors.append(GPU_COLOR)
        pos += 1

    pos += 0.5  # gap between variants

bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                showmeans=True, meanline=True,
                meanprops=dict(color="white", linewidth=1.5),
                medianprops=dict(color="yellow", linewidth=1.5),
                whiskerprops=dict(color="#e0e0e0"),
                capprops=dict(color="#e0e0e0"),
                flierprops=dict(markerfacecolor="#e0e0e0", markersize=4))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Latency (ms)")
ax.set_title("Inference Latency Distribution (22ch × 1280)", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, date_str, ha="right", fontsize=8, alpha=0.5)
plt.tight_layout()
fig.savefig(run_dir / "latency_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {run_dir}/latency_distribution.png")

# ── Summary table ────────────────────────────────────────────────────────

print(f"\n  ╔{'═'*60}╗")
print(f"  ║  {'Summary':^56}  ║")
print(f"  ╠{'═'*60}╣")
print(f"  ║  {'Variant':<12} {'Backend':<8} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}  ║")
print(f"  ╠{'─'*60}╣")
for d in cpu_data:
    v = d["variant"].capitalize()
    inf = d["inference"]
    print(f"  ║  {v:<12} {'CPU':<8} {inf['mean_ms']:>7.1f}  {inf['min_ms']:>7.1f}  {inf['max_ms']:>7.1f}  {inf['std_ms']:>7.1f}  ║")
if gpu_data:
    for d in gpu_data:
        v = d["variant"].capitalize()
        inf = d["inference"]
        print(f"  ║  {v:<12} {'GPU':<8} {inf['mean_ms']:>7.1f}  {inf['min_ms']:>7.1f}  {inf['max_ms']:>7.1f}  {inf['std_ms']:>7.1f}  ║")
print(f"  ╚{'═'*60}╝")

# ── Speedup ──────────────────────────────────────────────────────────────

if gpu_data:
    print(f"\n  GPU Speedup:")
    for cpu_d, gpu_d in zip(cpu_data, gpu_data):
        v = cpu_d["variant"].capitalize()
        speedup = cpu_d["inference"]["mean_ms"] / gpu_d["inference"]["mean_ms"]
        faster = "GPU" if speedup > 1 else "CPU"
        ratio = speedup if speedup > 1 else 1/speedup
        print(f"    LUNA-{v}: {faster} is {ratio:.1f}× faster")

PYEOF

echo ""
echo "━━━ Results saved to $RUN_DIR/ ━━━"
echo ""
ls -la "$RUN_DIR/"
echo ""
echo "Done."
