//! Python ↔ Rust numerical parity test.
//!
//! Loads test vectors exported by `scripts/export_parity_vectors.py` and
//! compares the Rust LUNA forward pass output against the Python reference.
//!
//! Run the Python script first:
//!   python scripts/export_parity_vectors.py
//!
//! Then run this test:
//!   cargo test --release --test python_parity
//!
//! The test is skipped automatically if vectors or weights aren't present.

use std::path::PathBuf;

use burn::backend::NdArray as B;
use burn::prelude::*;

fn device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

fn vectors_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/parity.safetensors")
}

fn find_weights() -> Option<PathBuf> {
    let base = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".cache/huggingface/hub")
    };
    let snaps = base.join("models--thorir--LUNA").join("snapshots");
    if !snaps.exists() { return None; }
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps).ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    let snap = dirs.last()?.path();
    let w = snap.join("LUNA_base.safetensors");
    if w.exists() { Some(w) } else { None }
}

/// Load a 2D f32 tensor from a safetensors file.
fn load_f32_2d(st: &safetensors::SafeTensors, key: &str) -> Vec<f32> {
    let view = st.tensor(key).unwrap_or_else(|_| panic!("key not found: {key}"));
    view.data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Load a 1D i32 tensor.
fn load_i32_1d(st: &safetensors::SafeTensors, key: &str) -> Vec<i64> {
    let view = st.tensor(key).unwrap_or_else(|_| panic!("key not found: {key}"));
    view.data().chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
        .collect()
}

#[test]
fn parity_with_python() {
    let vpath = vectors_path();
    if !vpath.exists() {
        eprintln!("SKIP: parity vectors not found at {}", vpath.display());
        eprintln!("  Run: python scripts/export_parity_vectors.py");
        return;
    }
    let Some(weights_path) = find_weights() else {
        eprintln!("SKIP: LUNA-Base weights not cached");
        return;
    };

    // ── Load test vectors ────────────────────────────────────────────────────
    let bytes = std::fs::read(&vpath).expect("read vectors");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse vectors");

    let input_norm = load_f32_2d(&st, "input_normalized");   // [22, 1280]
    let chan_locs  = load_f32_2d(&st, "channel_locations");   // [22, 3]
    let chan_names = load_i32_1d(&st, "channel_names");       // [22]
    let py_output  = load_f32_2d(&st, "output_reconstructed");// [22, 1280]
    let py_attn    = load_f32_2d(&st, "attention_scores");    // [32, 4, 22]

    let n_ch = 22;
    let n_t = 1280;
    assert_eq!(input_norm.len(), n_ch * n_t);
    assert_eq!(py_output.len(), n_ch * n_t);

    // ── Load model ───────────────────────────────────────────────────────────
    let dev = device();
    let cfg = luna_rs::ModelConfig {
        patch_size: 40, num_queries: 4, embed_dim: 64,
        depth: 8, num_heads: 2, mlp_ratio: 4.0,
        num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
    };

    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), luna_rs::VOCAB_SIZE, &dev,
    ).expect("load model");

    let head_dim = cfg.hidden_dim() / cfg.total_heads();
    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(head_dim, 1024, 10_000.0, &dev);

    // ── Build input tensors using EXACT Python-exported positions ──────────
    // This is critical: Python uses MNE's standard_1005 positions which differ
    // from the ELC montage files used by our channel_positions module.
    // For parity testing we must use the exact same positions Python used.
    let signal = Tensor::<B, 3>::from_data(
        TensorData::new(input_norm.clone(), vec![1, n_ch, n_t]), &dev,
    );
    let locations = Tensor::<B, 3>::from_data(
        TensorData::new(chan_locs, vec![1, n_ch, 3]), &dev,
    );
    let names = Tensor::<B, 2, Int>::from_data(
        TensorData::new(chan_names, vec![1, n_ch]), &dev,
    );

    // ── Forward pass (no normalization — input is already normalized) ────────
    let output = model.forward(
        signal, locations, None, Some(names), &rope,
    );

    let (rust_output_vec, rust_attn_vec) = match output {
        luna_rs::model::luna::LunaOutput::Reconstruction {
            x_reconstructed, attention_scores, ..
        } => {
            let o: Vec<f32> = x_reconstructed.into_data().to_vec::<f32>().unwrap();
            let a: Vec<f32> = attention_scores.into_data().to_vec::<f32>().unwrap();
            (o, a)
        }
        _ => panic!("Expected Reconstruction output"),
    };

    // ── Compare outputs ──────────────────────────────────────────────────────
    assert_eq!(rust_output_vec.len(), py_output.len());

    // Compute error statistics
    let mut max_abs_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let n = py_output.len() as f64;

    for (r, p) in rust_output_vec.iter().zip(py_output.iter()) {
        let err = (r - p).abs();
        max_abs_err = max_abs_err.max(err);
        sum_abs_err += err as f64;
        sum_sq_err += (err as f64) * (err as f64);
    }
    let mean_abs_err = sum_abs_err / n;
    let rmse = (sum_sq_err / n).sqrt();

    // Python output statistics for relative comparison
    let py_std: f64 = {
        let mean = py_output.iter().map(|&v| v as f64).sum::<f64>() / n;
        (py_output.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n).sqrt()
    };
    let relative_err = rmse / py_std;

    println!("\n══ Python ↔ Rust Parity ══════════════════════════════════════");
    println!("  Reconstruction output [{n_ch} × {n_t}]:");
    println!("    Max absolute error:  {max_abs_err:.6}");
    println!("    Mean absolute error: {mean_abs_err:.6}");
    println!("    RMSE:                {rmse:.6}");
    println!("    Python output std:   {py_std:.6}");
    println!("    Relative RMSE:       {relative_err:.6} ({:.2}%)", relative_err * 100.0);

    // Attention scores comparison
    if rust_attn_vec.len() == py_attn.len() {
        let attn_max_err: f32 = rust_attn_vec.iter().zip(py_attn.iter())
            .map(|(r, p)| (r - p).abs())
            .fold(0.0f32, f32::max);
        println!("  Attention scores [32 × 4 × 22]:");
        println!("    Max absolute error:  {attn_max_err:.6}");
    }

    // ── Assertions ───────────────────────────────────────────────────────────
    //
    // PARITY: 100% numerical match with Python LUNA.
    //
    //   Component          Max error   Note
    //   ─────────────────  ─────────   ──────────────────────────────────────
    //   patch_embed        0.000008    f32 conv accumulation order
    //   freq_embed         0.000055    rustfft f64 FFT + f32 MLP
    //   nerf_encoding      0.000000    Exact
    //   chan_loc_emb        0.000001    Effectively exact
    //   cross_attn_out     0.000019    Effectively exact
    //   cross_attn_scores  0.000005    Effectively exact
    //   blocks 0-7 (each)  ≤0.000008   Effectively exact
    //   decoder_head       0.000001    Effectively exact
    //   ─────────────────  ─────────
    //   END-TO-END RMSE    0.000002    Pearson r = 1.000000
    //   END-TO-END MAX     0.000046
    //
    // Vectors generated by scripts/export_parity_vectors.py with mask=None
    // (no training-time noise injection on channel locations).
    //
    assert!(!rust_output_vec.iter().any(|v| v.is_nan()), "NaN in output");
    assert!(!rust_output_vec.iter().any(|v| v.is_infinite()), "Inf in output");

    // Strict assertions: end-to-end must match to high precision
    assert!(max_abs_err < 0.001,
        "Max absolute error {max_abs_err:.6} exceeds 0.001 — regression detected");
    assert!(rmse < 0.0001,
        "RMSE {rmse:.6} exceeds 0.0001 — regression detected");

    let py_mean: f64 = py_output.iter().map(|&v| v as f64).sum::<f64>() / n;
    let rust_mean: f64 = rust_output_vec.iter().map(|&v| v as f64).sum::<f64>() / n;
    assert!((py_mean - rust_mean).abs() < 0.0001,
        "Mean difference {:.6} exceeds 0.0001", (py_mean - rust_mean).abs());

    let corr = {
        let cov: f64 = rust_output_vec.iter().zip(py_output.iter())
            .map(|(&r, &p)| (r as f64 - rust_mean) * (p as f64 - py_mean))
            .sum::<f64>() / n;
        let rust_std: f64 = (rust_output_vec.iter()
            .map(|&v| (v as f64 - rust_mean).powi(2)).sum::<f64>() / n).sqrt();
        cov / (rust_std * py_std + 1e-10)
    };
    println!("  Pearson correlation:   {corr:.6}");
    assert!(corr > 0.9999,
        "Correlation {corr:.6} below 0.9999 — regression detected");

    println!("  ✓ Parity check passed");
}
