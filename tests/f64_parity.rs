//! F64 precision parity test.
//!
//! Runs the same parity comparison as python_parity.rs but using NdArray<f64>
//! to eliminate f32 matmul accumulation order as an error source.

use std::path::PathBuf;
use burn::prelude::*;

type B = burn::backend::NdArray<f64>;
fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }

fn vectors_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/parity.safetensors")
}

fn find_weights() -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or(".".into());
    let snaps = PathBuf::from(&home).join(".cache/huggingface/hub/models--thorir--LUNA/snapshots");
    if !snaps.exists() { return None; }
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps).ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    let p = dirs.last()?.path().join("LUNA_base.safetensors");
    if p.exists() { Some(p) } else { None }
}

fn load_f32(st: &safetensors::SafeTensors, key: &str) -> Vec<f32> {
    st.tensor(key).unwrap().data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
fn load_i32_as_i64(st: &safetensors::SafeTensors, key: &str) -> Vec<i64> {
    st.tensor(key).unwrap().data().chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64).collect()
}

#[test]
fn parity_f64() {
    let vpath = vectors_path();
    if !vpath.exists() { eprintln!("SKIP: no parity vectors"); return; }
    let Some(wpath) = find_weights() else { eprintln!("SKIP: no weights"); return; };

    let bytes = std::fs::read(&vpath).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();

    let input_norm = load_f32(&st, "input_normalized");
    let chan_locs  = load_f32(&st, "channel_locations");
    let chan_names = load_i32_as_i64(&st, "channel_names");
    let py_output  = load_f32(&st, "output_reconstructed");

    let n_ch = 22; let n_t = 1280;
    let dev = device();
    let cfg = luna_rs::ModelConfig {
        patch_size: 40, num_queries: 4, embed_dim: 64,
        depth: 8, num_heads: 2, mlp_ratio: 4.0,
        num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
    };

    // Load model with f64 backend — weights auto-upcast from f32
    let model = luna_rs::weights::load_model::<B>(
        &cfg, wpath.to_str().unwrap(), luna_rs::VOCAB_SIZE, &dev,
    ).expect("load model f64");

    let head_dim = cfg.hidden_dim() / cfg.total_heads();
    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(head_dim, 1024, 10_000.0, &dev);

    // Build input — f32 data auto-upcasts to f64 in the backend
    let signal = Tensor::<B, 3>::from_data(
        TensorData::new(input_norm, vec![1, n_ch, n_t]), &dev,
    );
    let locations = Tensor::<B, 3>::from_data(
        TensorData::new(chan_locs, vec![1, n_ch, 3]), &dev,
    );
    let names = Tensor::<B, 2, Int>::from_data(
        TensorData::new(chan_names, vec![1, n_ch]), &dev,
    );

    let output = model.forward(signal, locations, None, Some(names), &rope);

    let rust_output: Vec<f64> = match output {
        luna_rs::model::luna::LunaOutput::Reconstruction { x_reconstructed, .. } => {
            x_reconstructed.into_data().to_vec::<f64>().unwrap()
        }
        _ => panic!("Expected Reconstruction"),
    };

    // Compare
    let n = py_output.len() as f64;
    let mut max_err: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for (r, p) in rust_output.iter().zip(py_output.iter()) {
        let err = (*r - *p as f64).abs();
        max_err = max_err.max(err);
        sum_sq += err * err;
    }
    let rmse = (sum_sq / n).sqrt();
    let py_std: f64 = {
        let mean = py_output.iter().map(|&v| v as f64).sum::<f64>() / n;
        (py_output.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
    };
    let py_mean: f64 = py_output.iter().map(|&v| v as f64).sum::<f64>() / n;
    let rust_mean: f64 = rust_output.iter().sum::<f64>() / n;

    let corr = {
        let cov: f64 = rust_output.iter().zip(py_output.iter())
            .map(|(&r, &p)| (r - rust_mean) * (p as f64 - py_mean))
            .sum::<f64>() / n;
        let rust_std = (rust_output.iter().map(|&v| (v - rust_mean).powi(2)).sum::<f64>() / n).sqrt();
        cov / (rust_std * py_std + 1e-10)
    };

    println!("\n══ F64 Parity ════════════════════════════════════════════════");
    println!("  Max absolute error:  {max_err:.6}");
    println!("  RMSE:                {rmse:.6}");
    println!("  Relative RMSE:       {:.6} ({:.2}%)", rmse / py_std, rmse / py_std * 100.0);
    println!("  Pearson correlation:   {corr:.6}");

    assert!(!rust_output.iter().any(|v| v.is_nan()), "NaN in f64 output");
}
