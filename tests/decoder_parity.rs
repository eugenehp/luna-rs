//! Decoder head parity test — isolate the reconstruction head.

use std::path::PathBuf;
use burn::backend::NdArray as B;
use burn::prelude::*;

fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }

fn load_f32(st: &safetensors::SafeTensors, key: &str) -> Vec<f32> {
    st.tensor(key).unwrap().data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn decoder_head_parity() {
    let dec_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/decoder.safetensors");
    let blk_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/blocks.safetensors");
    if !dec_path.exists() || !blk_path.exists() { eprintln!("SKIP"); return; }
    let wpath = {
        let home = std::env::var("HOME").unwrap_or(".".into());
        let p = PathBuf::from(&home).join(".cache/huggingface/hub/models--thorir--LUNA/snapshots");
        if !p.exists() { eprintln!("SKIP"); return; }
        let mut dirs: Vec<_> = std::fs::read_dir(&p).unwrap()
            .filter_map(|e| e.ok()).filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false)).collect();
        dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
        dirs.last().unwrap().path().join("LUNA_base.safetensors")
    };
    if !wpath.exists() { eprintln!("SKIP"); return; }

    let dev = device();
    let cfg = luna_rs::ModelConfig {
        patch_size: 40, num_queries: 4, embed_dim: 64,
        depth: 8, num_heads: 2, mlp_ratio: 4.0,
        num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
    };
    let model = luna_rs::weights::load_model::<B>(
        &cfg, wpath.to_str().unwrap(), 90, &dev,
    ).unwrap();

    let blk_bytes = std::fs::read(&blk_path).unwrap();
    let blk_st = safetensors::SafeTensors::deserialize(&blk_bytes).unwrap();
    let dec_bytes = std::fs::read(&dec_path).unwrap();
    let dec_st = safetensors::SafeTensors::deserialize(&dec_bytes).unwrap();

    // Load Python inputs to the decoder
    let py_norm_out = load_f32(&blk_st, "norm_out");            // [32, 256]
    let py_dec_queries = load_f32(&dec_st, "decoder_queries");    // [32, 22, 64]
    let py_recon = load_f32(&dec_st, "x_reconstructed");          // [22, 1280]

    // Build Rust tensors
    let x_latent = Tensor::<B, 3>::from_data(
        TensorData::new(py_norm_out, vec![1, 32, 256]), &dev,
    );
    let decoder_queries = Tensor::<B, 3>::from_data(
        TensorData::new(py_dec_queries, vec![32, 22, 64]), &dev,
    );

    // Run decoder head
    let rust_recon = model.decoder_head.as_ref().unwrap().forward(
        x_latent, decoder_queries, 22,
    );
    let rust_vec: Vec<f32> = rust_recon.squeeze::<2>().into_data().to_vec::<f32>().unwrap();

    let max_err: f32 = rust_vec.iter().zip(py_recon.iter())
        .map(|(r, p)| (r - p).abs()).fold(0.0f32, f32::max);
    let mean_err: f64 = rust_vec.iter().zip(py_recon.iter())
        .map(|(r, p)| (r - p).abs() as f64).sum::<f64>() / rust_vec.len() as f64;
    let rmse: f64 = (rust_vec.iter().zip(py_recon.iter())
        .map(|(r, p)| ((r - p) as f64).powi(2)).sum::<f64>() / rust_vec.len() as f64).sqrt();
    let py_std: f64 = {
        let n = py_recon.len() as f64;
        let mean = py_recon.iter().map(|&v| v as f64).sum::<f64>() / n;
        (py_recon.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
    };

    println!("\n══ Decoder Head Parity ════════════════════════════════════════");
    println!("  Max absolute error:  {max_err:.6}");
    println!("  Mean absolute error: {mean_err:.8}");
    println!("  RMSE:                {rmse:.6}");
    println!("  Relative RMSE:       {:.6} ({:.2}%)", rmse / py_std, rmse / py_std * 100.0);
}
