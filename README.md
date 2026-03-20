# luna-rs

**LUNA** (Latent Unified Network Architecture) EEG Foundation Model — inference in Rust with [Burn ML](https://burn.dev).

A pure-Rust implementation of the [LUNA](https://huggingface.co/thorir/LUNA) model from [BioFoundation](https://github.com/pulp-bio/BioFoundation) (ETH Zurich), a topology-agnostic EEG foundation model that uses cross-attention with learned queries to handle variable-channel EEG recordings.

Weights are downloaded automatically from HuggingFace. Numerical parity with the Python implementation is verified to **RMSE 0.000002** (Pearson r = 1.000000).

## Architecture

LUNA's key innovation is **channel unification via cross-attention**: regardless of whether the input has 20, 22, or 62 EEG channels, it compresses them into a fixed number of learned queries per time patch.

```
EEG signal (B, C, T)
    │
    ├─→ PatchEmbedNetwork (3-layer CNN)  ──┐
    │                                       ├─→ sum → (B, C×S, D)
    └─→ FrequencyFeatureEmbedder (FFT+MLP)─┘
                                            │
                              + NeRF positional encoding of 3D electrode locations
                              + channel location MLP
                              + mask tokens (if pre-training)
                                            │
                              rearrange: (B, C×S, D) → (B×S, C, D)
                                            │
                              CrossAttentionBlock
                              Q learned queries attend to C channels
                              → FFN → 3-layer query self-attention
                                            │
                              (B×S, Q, D) → reshape → (B, S, Q×D)
                                            │
                              N × RotaryTransformerBlock (RoPE self-attention + FFN)
                                            │
                              LayerNorm → (B, S, Q×D)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │ Reconstruction (pretrain)                      │ Classification (finetune)
                    │                                                │
        TransformerDecoderLayer                           Learned aggregation query
        (channel queries reconstruct patches)             → cross-attention → MLP
                    │                                                │
              (B, C, T) signal                              (B, num_classes) logits
```

### Model Variants

| Variant    | Params | Layers | Queries (Q) | embed_dim (D) | Q×D |
|------------|--------|--------|-------------|---------------|-----|
| LUNA-Base  | 7M     | 8      | 4           | 64            | 256 |
| LUNA-Large | 43M    | 10     | 6           | 96            | 576 |
| LUNA-Huge  | 311M   | 24     | 8           | 128           | 1024|

Weights hosted at [`thorir/LUNA`](https://huggingface.co/thorir/LUNA) on HuggingFace.

---

## Quick Start

```sh
# Download weights and run reconstruction on synthetic EEG
cargo run --example reconstruct --release --features hf-download -- -v
```

Output:
```
▸ Input: 22 channels × 1280 samples (5s @ 256Hz)
▸ Forward pass: 83 ms

▸ Outputs:
  x_reconstructed: [1, 22, 1280]
  attention_scores: [32, 4, 22]

▸ Query → Channel attention (first time patch):
    Q0: top-3 = P3-O1(0.565), P4-O2(0.193), T3-C3(0.177)
    Q1: top-3 = CZ-C4(0.242), C3-CZ(0.239), F3-C3(0.212)
    Q2: top-3 = C4-P4(0.371), T4-A2(0.336), A1-T3(0.126)
    Q3: top-3 = F7-T3(0.454), FP2-F8(0.231), T4-T6(0.112)
```

---

## Build

```sh
# CPU (default — Rayon multi-threading + SIMD)
cargo build --release

# CPU — macOS with Apple Accelerate BLAS
cargo build --release --features blas-accelerate

# CPU — Linux with OpenBLAS
cargo build --release --features openblas-system

# GPU — cross-platform WGSL shaders (Metal on macOS, Vulkan on Linux, DX12 on Windows)
cargo build --release --no-default-features --features wgpu

# GPU — macOS: native Metal shaders (MSL) — fastest on Apple Silicon
cargo build --release --no-default-features --features metal

# GPU — Linux/Windows: native Vulkan shaders (SPIR-V) — fastest on NVIDIA/AMD
cargo build --release --no-default-features --features vulkan
```

### GPU Backend Details

| Platform | Runtime | Shader pipeline | Feature flag |
|----------|---------|----------------|--------------|
| macOS    | Metal   | WGSL (generic) | `--features wgpu` |
| macOS    | Metal   | MSL (native, faster) | `--features metal` |
| Linux    | Vulkan  | WGSL (generic) | `--features wgpu` |
| Linux    | Vulkan  | SPIR-V (native, faster) | `--features vulkan` |
| Windows  | Vulkan/DX12 | WGSL (generic) | `--features wgpu` |
| Windows  | Vulkan  | SPIR-V (native, faster) | `--features vulkan` |

---

## API

### High-level: `LunaEncoder`

```rust
use luna_rs::{LunaEncoder, build_batch_named, TUEG_CHANNELS};
use std::path::Path;

// Load model
let (encoder, _ms) = LunaEncoder::<B>::load(
    Path::new("config.json"),
    Path::new("model.safetensors"),
    device,
)?;

// Build input from channel names (auto-resolves positions + vocab indices)
let batch = build_batch_named::<B>(signal_vec, TUEG_CHANNELS, 1280, &device);

// Run inference
let result = encoder.run_batch(&batch)?;
println!("Output shape: {:?}", result.shape);

// Save / load results
result.save_safetensors("output.safetensors")?;
let loaded = EncodingResult::load_safetensors("output.safetensors")?;
```

### Low-level: direct model access

```rust
use luna_rs::model::luna::{Luna, LunaOutput};
use luna_rs::model::rope::RotaryEmbedding;

let model = luna_rs::weights::load_model::<B>(&cfg, "weights.safetensors", 90, &device)?;
let rope = RotaryEmbedding::new(head_dim, 1024, 10_000.0, &device);

let output = model.forward(signal, channel_locations, None, Some(channel_names), &rope);

match output {
    LunaOutput::Reconstruction { x_reconstructed, x_original, attention_scores } => { ... }
    LunaOutput::Classification { logits, x_original } => { ... }
}
```

### CSV input

```rust
use luna_rs::load_from_csv;

let (batches, info) = load_from_csv::<B>(Path::new("recording.csv"), 256.0, 1280, &device)?;
println!("{} epochs from {} channels", info.n_epochs, info.ch_names.len());
```

---

## Examples

All examples auto-download LUNA-Base weights from HuggingFace.

| Example | What it demonstrates | Command |
|---------|---------------------|---------|
| **`load_and_inspect`** | Download weights, print architecture summary and parameter breakdown | `cargo run --example load_and_inspect --release --features hf-download` |
| **`reconstruct`** | Full reconstruction forward pass, per-channel RMSE, query→channel attention patterns | `cargo run --example reconstruct --release --features hf-download -- -v` |
| **`channel_invariance`** | Same model on 4 different channel counts (8, 10, 16, 22) — all work | `cargo run --example channel_invariance --release --features hf-download` |
| **`benchmark`** | Inference latency, channel-scaling benchmark (4→32 channels) | `cargo run --example benchmark --release --features hf-download` |
| **`embed`** | High-level `LunaEncoder` API, multi-epoch processing, save to safetensors | `cargo run --example embed --release --features hf-download -- -v` |

Use `--variant large` or `--variant huge` to switch model sizes.

---

## Binaries

| Binary | Purpose | Command |
|--------|---------|---------|
| **`infer`** | Run inference on dummy input, print timing | `cargo run --release -- --weights W --config C --output O` |
| **`download_weights`** | Download weights from HuggingFace | `cargo run --bin download_weights --release --features hf-download -- --variant base` |

---

## Python Parity

Numerically verified against the Python [BioFoundation](https://github.com/pulp-bio/BioFoundation) LUNA implementation. Test vectors are exported from Python with `mask=None` (inference mode) and compared in Rust with strict assertions.

### Per-component accuracy

| Component | Max error | Test file |
|-----------|-----------|-----------|
| `PatchEmbedNetwork` (3-layer CNN) | 0.000008 | `intermediate_parity.rs` |
| `FrequencyFeatureEmbedder` (rustfft f64 + MLP) | 0.000055 | `intermediate_parity.rs` |
| `nerf_positional_encoding` | 0.000000 | `intermediate_parity.rs` |
| `channel_location_embedder` (MLP) | 0.000001 | `intermediate_parity.rs` |
| `CrossAttentionBlock` output | 0.000019 | `intermediate_parity.rs` |
| `CrossAttentionBlock` attention scores | 0.000005 | `intermediate_parity.rs` |
| Transformer blocks 0–7 (each) | ≤ 0.000008 | `block_parity.rs` |
| `ReconstructionHead` (TransformerDecoder) | 0.000003 | `decoder_parity.rs` |

### End-to-end accuracy

| Metric | Value |
|--------|-------|
| **RMSE** | **0.000002** |
| **Max absolute error** | **0.000046** |
| **Relative RMSE** | **0.000005 (0.00%)** |
| **Pearson correlation** | **1.000000** |

### Reproducing parity tests

```sh
# 1. Export Python reference vectors (requires PyTorch + BioFoundation repo)
python scripts/export_parity_vectors.py
python scripts/export_intermediates.py

# 2. Run all 24 tests
cargo test --release
```

### What enables exact parity

| Technique | Why it matters |
|-----------|---------------|
| `rustfft` in **f64** for FFT | Matches `torch.fft.rfft`'s internal f64 promotion on CPU |
| `f32::atan2` on CPU | Bit-identical to PyTorch's `torch.angle()` (same libc `atan2f`) |
| `FusedMultiheadAttention` with single `in_proj` Linear | Matches `nn.MultiheadAttention`'s fused `in_proj_weight [3D, D]` layout |
| `TransformerEncoderLayer` with `norm_first` | Matches `nn.TransformerEncoderLayer(norm_first=True)` structure |
| 3-sublayer `TransformerDecoderLayer` | Self-attn → cross-attn → FFN, matches `nn.TransformerDecoderLayer(norm_first=True)` |
| `mask=None` at inference | Avoids Python's training-time `randn * 0.02` noise on channel locations |
| Correct `(D E)` flatten in `PatchEmbedNetwork` | Matches `einops.rearrange('B E CS D -> B CS (D E)')` — D-inner, E-outer |
| `repeat_dim(0, n)` for channel embeddings | Matches PyTorch `.repeat(n, 1, 1)` tile semantics |
| DC/Nyquist bin clamping in FFT | Forces `imag=0` at k=0 and k=N/2, matching `rfft` guarantees |

---

## Test Suite

24 tests across 8 test files, all passing with zero warnings.

| File | Tests | What it verifies |
|------|-------|------------------|
| `tests/python_parity.rs` | 1 | End-to-end: RMSE < 0.0001, correlation > 0.9999 |
| `tests/intermediate_parity.rs` | 1 | Per-component: patch, freq, nerf, loc, cross-attn (all < 0.000055) |
| `tests/block_parity.rs` | 1 | Per-transformer-block: 8 blocks + norm (all < 0.000008) |
| `tests/decoder_parity.rs` | 1 | Decoder head in isolation (max_err = 0.000003) |
| `tests/f64_parity.rs` | 1 | f64 backend gives same parity (RMSE = 0.000002) |
| `tests/forward_pass.rs` | 4 | Output shapes, value ranges, variable channels (4–29), channel vocab |
| `src/lib.rs` (unit) | 15 | Channel vocab (7), positions (3), CSV (2), conv2d (1), patch_embed (1), repeat_dim (1) |

---

## Project Structure

```
luna-rs/
├── src/
│   ├── lib.rs                  # Public API, re-exports
│   ├── config.rs               # ModelConfig, DataConfig
│   ├── data.rs                 # InputBatch, build_batch, build_batch_named, channel_wise_normalize
│   ├── encoder.rs              # LunaEncoder (high-level API), EncodingResult (save/load safetensors)
│   ├── weights.rs              # WeightMap, load_model (safetensors → Burn tensors)
│   ├── channel_positions.rs    # 6 embedded ELC montage files, bipolar_channel_xyz
│   ├── channel_vocab.rs        # 90-channel vocabulary (TUEG + Siena + SEED)
│   ├── csv_loader.rs           # load_from_csv (CSV → InputBatch epochs)
│   ├── model/
│   │   ├── luna.rs             # Full LUNA model, nerf_positional_encoding, LunaOutput enum
│   │   ├── patch_embed.rs      # PatchEmbedNetwork (3-layer CNN)
│   │   ├── freq_embed.rs       # FrequencyFeatureEmbedder (rustfft f64 + MLP)
│   │   ├── cross_attention.rs  # CrossAttentionBlock, FusedMultiheadAttention, TransformerEncoderLayer
│   │   ├── attention.rs        # RotarySelfAttention (1-D RoPE)
│   │   ├── encoder_block.rs    # RotaryEncoderBlock (norm → attn → norm → FFN)
│   │   ├── feedforward.rs      # FeedForward (fc1 → GELU → LayerNorm → fc2)
│   │   ├── rope.rs             # RotaryEmbedding (precomputed rotation matrices)
│   │   ├── norm.rs             # LunaLayerNorm wrapper
│   │   ├── reconstruction_head.rs  # PatchReconstructionHead (TransformerDecoderLayer + MLP)
│   │   └── classification_head.rs  # ClassificationHead (aggregation query + MLP)
│   ├── bin/
│   │   ├── infer.rs            # CLI inference binary
│   │   └── download_weights.rs # HuggingFace weight downloader
│   └── montages/               # 6 ASA .elc montage files (standard_1005, 1020, etc.)
├── examples/
│   ├── common/mod.rs           # Shared utilities, HF weight resolution, synthetic EEG generation
│   ├── load_and_inspect.rs     # Architecture inspection
│   ├── reconstruct.rs          # Masked reconstruction with attention analysis
│   ├── channel_invariance.rs   # Variable channel count demonstration
│   ├── benchmark.rs            # Latency benchmarking
│   └── embed.rs                # High-level embedding extraction
├── tests/
│   ├── python_parity.rs        # End-to-end numerical parity (RMSE = 0.000002)
│   ├── intermediate_parity.rs  # Per-component numerical parity
│   ├── block_parity.rs         # Per-transformer-block parity
│   ├── decoder_parity.rs       # Decoder head parity
│   ├── f64_parity.rs           # f64 backend parity
│   ├── forward_pass.rs         # Integration tests with real weights
│   └── vectors/                # Exported Python reference tensors (safetensors)
├── scripts/
│   ├── export_parity_vectors.py     # Export Python LUNA output for Rust comparison
│   └── export_intermediates.py      # Export per-component intermediate tensors
├── Cargo.toml
├── README.md
└── PLAN.md                     # Development roadmap
```

---

## Dependencies

### Core (always compiled)
- [`burn`](https://burn.dev) 0.20.1 — ML framework (tensor ops, nn modules)
- [`rustfft`](https://crates.io/crates/rustfft) 6 — FFT for frequency embedder (exact parity with torch.fft.rfft)
- [`exg`](https://github.com/eugenehp/exg) — EEG preprocessing (FIF/EDF reader, filtering, resampling, montage)
- `safetensors` — weight loading and result I/O
- `serde` + `serde_json` — config parsing
- `half` — bf16→f32 weight conversion
- `anyhow` — error handling

### Optional
- `burn-ndarray` — CPU backend (default)
- `burn-wgpu` — GPU backend
- `hf-hub` — HuggingFace weight download (`--features hf-download`)
- `clap` — CLI argument parsing (binaries only)

---

## Citation

If you use LUNA, please cite the original paper:

```bibtex
@inproceedings{
  doner2025luna,
  title={{LUNA}: Efficient and Topology-Agnostic Foundation Model for {EEG} Signal Analysis},
  author={Berkay D{\"o}ner and Thorir Mar Ingolfsson and Luca Benini and Yawei Li},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=uazfjnFL0G}
}
```

## License

Apache-2.0
