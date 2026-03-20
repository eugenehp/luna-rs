//! Per-transformer-block parity test.
//! Run: python scripts/... && cargo test --release --test block_parity -- --nocapture

use std::path::PathBuf;
use burn::backend::NdArray as B;
use burn::prelude::*;

fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }

fn load_f32(st: &safetensors::SafeTensors, key: &str) -> Vec<f32> {
    st.tensor(key).unwrap().data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn per_block_parity() {
    let blk_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/blocks.safetensors");
    if !blk_path.exists() { eprintln!("SKIP: no block vectors"); return; }
    let wpath = {
        let home = std::env::var("HOME").unwrap_or(".".into());
        let snaps = PathBuf::from(&home).join(".cache/huggingface/hub/models--thorir--LUNA/snapshots");
        if !snaps.exists() { eprintln!("SKIP"); return; }
        let mut dirs: Vec<_> = std::fs::read_dir(&snaps).unwrap()
            .filter_map(|e| e.ok()).filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false)).collect();
        dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
        dirs.last().unwrap().path().join("LUNA_base.safetensors")
    };
    if !wpath.exists() { eprintln!("SKIP: no weights"); return; }

    let bytes = std::fs::read(&blk_path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let dev = device();

    let cfg = luna_rs::ModelConfig {
        patch_size: 40, num_queries: 4, embed_dim: 64,
        depth: 8, num_heads: 2, mlp_ratio: 4.0,
        num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
    };
    let model = luna_rs::weights::load_model::<B>(
        &cfg, wpath.to_str().unwrap(), 90, &dev,
    ).unwrap();
    let head_dim = cfg.hidden_dim() / cfg.total_heads();
    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(head_dim, 1024, 10_000.0, &dev);

    // Load Python's block input: [32, 256]
    let py_input = load_f32(&st, "block_input");
    let mut x = Tensor::<B, 3>::from_data(
        TensorData::new(py_input.clone(), vec![1, 32, 256]), &dev,
    );
    let freqs = rope.get_freqs(32);

    println!("\n══ Per-Block Parity ═══════════════════════════════════════════");
    for i in 0..8 {
        let py_out = load_f32(&st, &format!("block_{i}_out"));
        x = model.blocks[i].forward(x, freqs.clone());
        let rust_out: Vec<f32> = x.clone().squeeze::<2>().into_data().to_vec::<f32>().unwrap();

        let max_err: f32 = rust_out.iter().zip(py_out.iter())
            .map(|(r, p)| (r - p).abs()).fold(0.0f32, f32::max);
        let mean_err: f64 = rust_out.iter().zip(py_out.iter())
            .map(|(r, p)| (r - p).abs() as f64).sum::<f64>() / rust_out.len() as f64;
        println!("  block_{i}:  max_err={max_err:.6}  mean_err={mean_err:.8}");
    }

    // Final norm
    let py_norm = load_f32(&st, "norm_out");
    let rust_norm: Vec<f32> = model.norm.forward(x).squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    let max_err: f32 = rust_norm.iter().zip(py_norm.iter())
        .map(|(r, p)| (r - p).abs()).fold(0.0f32, f32::max);
    let mean_err: f64 = rust_norm.iter().zip(py_norm.iter())
        .map(|(r, p)| (r - p).abs() as f64).sum::<f64>() / rust_norm.len() as f64;
    println!("  norm:      max_err={max_err:.6}  mean_err={mean_err:.8}");
}
