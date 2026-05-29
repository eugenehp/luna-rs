# luna-rs

**LUNA** (Latent Unified Network Architecture) — EEG foundation model inference in Rust.

Pure-Rust port of [LUNA](https://huggingface.co/thorir/LUNA) ([BioFoundation](https://github.com/pulp-bio/BioFoundation)). Two inference engines are available:

| Engine | Feature | Role |
|--------|---------|------|
| **RLX** (default) | `rlx`, `rlx-cpu` | Fast compiled graph runtime (CPU, Metal, MLX, …) |
| **Burn** | `burn`, `ndarray` | Reference implementation, Python parity, GPU via wgpu |

LUNA compresses variable-channel EEG into fixed learned queries via cross-attention, then runs a rotary Transformer encoder.

Weights: [`thorir/LUNA`](https://huggingface.co/thorir/LUNA) on HuggingFace (auto-resolved from cache when present).

---

## Quick start (RLX)

```sh
# Default build: RLX on CPU
cargo build --release

# Inference on synthetic EEG (set paths or use HF cache)
export LUNA_WEIGHTS=~/.cache/huggingface/hub/models--thorir--LUNA/snapshots/<hash>/LUNA_base.safetensors
export LUNA_CONFIG=tests/vectors/config.json

cargo run --release --bin infer -- \
  --weights "$LUNA_WEIGHTS" --config "$LUNA_CONFIG" --output out.safetensors -v
```

Apple Silicon GPU:

```sh
cargo run --release --no-default-features \
  --features rlx,rlx-mlx --bin infer -- \
  --device mlx --weights "$LUNA_WEIGHTS" --config "$LUNA_CONFIG" --output out.safetensors
```

---

## Build

```sh
# RLX CPU (default)
cargo build --release

# RLX + Apple backends
cargo build --release --no-default-features --features rlx,rlx-cpu,rlx-metal,rlx-mlx

# Burn reference (NdArray CPU)
cargo build --release --no-default-features --features burn,ndarray

# Burn GPU (wgpu / Metal / Vulkan)
cargo build --release --no-default-features --features burn,wgpu
cargo build --release --no-default-features --features burn,metal   # macOS MSL
cargo build --release --no-default-features --features burn,vulkan # Linux SPIR-V

# Both engines (parity tests, backend_compare)
cargo build --release --no-default-features \
  --features burn,rlx,ndarray,rlx-cpu,rlx-metal,rlx-mlx
```

| RLX feature | Backend |
|-------------|---------|
| `rlx-cpu` | Multi-threaded CPU (Rayon) |
| `rlx-metal` | Apple Metal |
| `rlx-mlx` | Apple MLX |
| `rlx-gpu` / `rlx-cuda` / `rlx-tpu` | Other RLX targets (platform-dependent) |

| Burn feature | Backend |
|--------------|---------|
| `ndarray` | CPU (required with `burn`) |
| `wgpu` | Cross-platform GPU |
| `metal` / `vulkan` | Native shader pipelines |
| `hf-download` | HuggingFace weight download in examples |

---

## Benchmarks

**Burn vs RLX** (all compiled backends, optional parity vs Burn CPU):

```sh
cargo run --example backend_compare --release \
  --no-default-features \
  --features burn,rlx,ndarray,rlx-cpu,rlx-metal,rlx-mlx -- --parity
```

**Burn-only** latency across model variants (base / large / huge):

```sh
cargo run --example benchmark --release --features burn,hf-download
# or: ./bench.sh base,large,huge
```

Historical multi-platform charts live under `figures/` (from `bench.sh`).

---

## Architecture

```
EEG (B, C, T)
    ├─ PatchEmbedNetwork (3-layer CNN)
    └─ FrequencyFeatureEmbedder (FFT + MLP)
              → sum → + NeRF positions + channel MLP
              → CrossAttentionBlock (Q queries × C channels)
              → N × RotaryTransformerBlock
              → Reconstruction head or classification head
```

| Variant | Params | Layers | Q | D | Q×D |
|---------|--------|--------|---|---|-----|
| Base | 7M | 8 | 4 | 64 | 256 |
| Large | 43M | 10 | 6 | 96 | 576 |
| Huge | 311M | 24 | 8 | 128 | 1024 |

---

## Tests

```sh
# RLX smoke + parity (needs cached LUNA-Base weights)
cargo test --release --no-default-features \
  --features burn,rlx,ndarray,rlx-cpu \
  --test rlx_graph_compile --test parity_rlx_vs_burn \
  --test rlx_prepare_burn --test rlx_graph_parity

# Burn ↔ Python vectors (export first)
python scripts/export_parity_vectors.py
cargo test --release --no-default-features --features burn,ndarray \
  --test python_parity --test intermediate_parity --test block_parity \
  --test decoder_parity --test f64_parity --test forward_pass
```

| Test | Features | What it checks |
|------|----------|----------------|
| `rlx_graph_compile` | `rlx` | Graph compiles and runs (reconstruction + classification) |
| `parity_rlx_vs_burn` | `burn`, `rlx` | End-to-end RLX vs Burn (max_abs ≲ 3e-6) |
| `rlx_graph_parity` | `burn`, `rlx` | RLX graph vs Burn with shared prepared tokens |
| `rlx_prepare_burn` | `burn`, `rlx` | CPU prepare path vs Burn intermediates |
| `parity_rlx_vs_python` | `rlx` | RLX vs exported Python vectors |
| `python_parity` | `burn` | Burn vs Python (RMSE ≈ 2e-6) |
| `intermediate_parity` | `burn` | Per-component vs Python |
| `block_parity` / `decoder_parity` | `burn` | Block-wise / decoder parity |
| `forward_pass` | `burn` | Shapes and ranges with real weights |

Set `LUNA_WEIGHTS` or cache `thorir/LUNA` locally. RLX device override: `LUNA_RLX_DEVICE=cpu|metal|mlx`.

Shared helpers: `tests/common/mod.rs`.

---

## Examples (Burn)

Requires `--features burn` (and usually `hf-download`):

| Example | Command |
|---------|---------|
| `load_and_inspect` | `cargo run --example load_and_inspect --release --features burn,hf-download` |
| `reconstruct` | `cargo run --example reconstruct --release --features burn,hf-download -- -v` |
| `channel_invariance` | `cargo run --example channel_invariance --release --features burn,hf-download` |
| `embed` | `cargo run --example embed --release --features burn,hf-download` |
| `benchmark` | `cargo run --example benchmark --release --features burn,hf-download` |

Use `--variant large` or `huge` where supported.

---

## Binaries

| Binary | Features | Purpose |
|--------|----------|---------|
| `infer` | `rlx` (default) | RLX inference CLI (EDF/FIF or synthetic) |
| `download_weights` | `hf-download` | Download from HuggingFace |
| `gen_sample_eeg` | — | Synthetic CSV for testing |
| `safetensors_info` | — | Inspect tensor keys in a checkpoint |

---

## API

**RLX (default crate root when `burn` is off):**

```rust
use luna_rs::{LunaEncoder, init_threads};
use std::path::Path;

let _ = init_threads(None);
let (mut enc, _) = LunaEncoder::load(
    Path::new("config.json"),
    Path::new("LUNA_base.safetensors"),
    rlx::Device::Cpu,
)?;
let out = enc.run_epoch(&signal, &locations, Some(&names), n_ch, n_t)?;
```

**Burn (`--features burn`):**

```rust
use luna_rs::{LunaEncoder, build_batch_named, TUEG_CHANNELS};
// LunaEncoder<B>, model::luna::Luna, load_from_csv, …
```

---

## Project layout

```
src/
  lib.rs, config.rs, channel_*.rs
  rlx/          # RLX graph, prepare, encoder (default path)
  model/        # Burn modules
  bin/          # infer, download_weights, …
examples/
  backend_compare.rs   # Burn vs RLX benchmark
  reconstruct.rs, …    # Burn demos
tests/
  common/              # Shared weight paths, synthetic data
  *_parity.rs          # Numerical tests
scripts/               # Python vector export
```

---

## Parity notes

- **Burn ↔ Python**: RMSE ≈ 2e-6 with exported vectors (`mask=None` inference).
- **RLX ↔ Burn**: max_abs ≲ 3e-6 on LUNA-Base, 22 ch × 1280 samples.
- **RLX ↔ Python**: requires `scripts/export_parity_vectors.py` (BioFoundation).
- **MLX**: attention uses `[B,H,S,D]` layout in the RLX graph (required by MLX SDPA).

---

## Citation

```bibtex
@inproceedings{
  doner2025luna,
  title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
  author={Berkay D{\"o}ner and Thorir Mar Ingolfsson and Luca Benini and Yawei Li},
  booktitle={NeurIPS},
  year={2025},
  url={https://openreview.net/forum?id=uazfjnFL0G}
}
```

## License

Apache-2.0
