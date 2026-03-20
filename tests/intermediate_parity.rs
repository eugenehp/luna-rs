//! Compare intermediate tensor values against Python exports.
//! Run: python scripts/export_intermediates.py && cargo test --release --test intermediate_parity

use std::path::PathBuf;
use burn::backend::NdArray as B;
use burn::prelude::*;

fn device() -> burn::backend::ndarray::NdArrayDevice { burn::backend::ndarray::NdArrayDevice::Cpu }

fn load_f32(st: &safetensors::SafeTensors, key: &str) -> (Vec<f32>, Vec<usize>) {
    let v = st.tensor(key).unwrap();
    let data: Vec<f32> = v.data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    (data, v.shape().to_vec())
}

fn compare(name: &str, rust: &[f32], python: &[f32]) {
    assert_eq!(rust.len(), python.len(), "{name} length mismatch");
    let max_err: f32 = rust.iter().zip(python.iter())
        .map(|(r, p)| (r - p).abs()).fold(0.0f32, f32::max);
    let mean_err: f64 = rust.iter().zip(python.iter())
        .map(|(r, p)| (r - p).abs() as f64).sum::<f64>() / rust.len() as f64;
    println!("  {name:25}  max_err={max_err:.6}  mean_err={mean_err:.8}");
}

#[test]
fn compare_intermediates() {
    let int_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/intermediates.safetensors");
    let par_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/parity.safetensors");
    if !int_path.exists() || !par_path.exists() {
        eprintln!("SKIP: run python scripts/export_intermediates.py first");
        return;
    }
    let weights_path = {
        let home = std::env::var("HOME").unwrap_or(".".into());
        let snaps = PathBuf::from(&home).join(".cache/huggingface/hub/models--thorir--LUNA/snapshots");
        if !snaps.exists() { eprintln!("SKIP: no weights"); return; }
        let mut dirs: Vec<_> = std::fs::read_dir(&snaps).unwrap()
            .filter_map(|e| e.ok()).filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false)).collect();
        dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
        let p = dirs.last().unwrap().path().join("LUNA_base.safetensors");
        if !p.exists() { eprintln!("SKIP: no weights"); return; }
        p
    };

    let int_bytes = std::fs::read(&int_path).unwrap();
    let int_st = safetensors::SafeTensors::deserialize(&int_bytes).unwrap();
    let par_bytes = std::fs::read(&par_path).unwrap();
    let par_st = safetensors::SafeTensors::deserialize(&par_bytes).unwrap();

    let dev = device();
    let cfg = luna_rs::ModelConfig {
        patch_size: 40, num_queries: 4, embed_dim: 64,
        depth: 8, num_heads: 2, mlp_ratio: 4.0,
        num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
    };
    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), 90, &dev,
    ).unwrap();

    // Load Python input
    let (input_data, _) = load_f32(&par_st, "input_normalized");
    let (locs_data, _) = load_f32(&par_st, "channel_locations");
    let n_ch = 22; let n_t = 1280;

    let input_t = Tensor::<B, 3>::from_data(TensorData::new(input_data, vec![1, n_ch, n_t]), &dev);
    let locs_t = Tensor::<B, 3>::from_data(TensorData::new(locs_data, vec![1, n_ch, 3]), &dev);

    println!("\n══ Intermediate Parity ════════════════════════════════════════");

    // 1. Patch embed
    let (py_pe, _) = load_f32(&int_st, "patch_embed_output");
    let rust_pe: Vec<f32> = model.patch_embed.forward(input_t.clone())
        .squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    compare("patch_embed", &rust_pe, &py_pe);

    // 2. Freq embed
    let (py_fe, _) = load_f32(&int_st, "freq_embed_output");
    let rust_fe: Vec<f32> = model.freq_embed.forward(input_t.clone())
        .squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    compare("freq_embed", &rust_fe, &py_fe);

    // 3. Normalized locations
    let (py_nlocs, _) = load_f32(&int_st, "normed_locs");
    let normed = {
        let mins = locs_t.clone().min_dim(1);
        let maxs = locs_t.clone().max_dim(1);
        (locs_t.clone() - mins.clone()) / (maxs - mins + 1e-8)
    };
    let rust_nlocs: Vec<f32> = normed.clone().squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    compare("normed_locs", &rust_nlocs, &py_nlocs);

    // 4. NeRF encoding
    let (py_nerf, _) = load_f32(&int_st, "nerf_encoded");
    let rust_nerf_t = luna_rs::model::luna::nerf_positional_encoding(normed.clone(), 64, &dev);
    let rust_nerf: Vec<f32> = rust_nerf_t.clone().squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    compare("nerf_encoded", &rust_nerf, &py_nerf);

    // 5. Channel location embedder
    let (py_cle, _) = load_f32(&int_st, "chan_loc_emb");
    let rust_cle = {
        let h = burn::tensor::activation::gelu(model.chan_loc_fc1.forward(rust_nerf_t.clone()));
        let h = model.chan_loc_norm.forward(h);
        model.chan_loc_fc2.forward(h)
    };
    let rust_cle_vec: Vec<f32> = rust_cle.clone().squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    compare("chan_loc_emb", &rust_cle_vec, &py_cle);

    // ── Deeper intermediates (from intermediates2.safetensors) ────────────────
    let int2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/intermediates2.safetensors");
    if !int2_path.exists() {
        eprintln!("  (skipping deeper intermediates — run export script)");
        return;
    }
    let int2_bytes = std::fs::read(&int2_path).unwrap();
    let int2_st = safetensors::SafeTensors::deserialize(&int2_bytes).unwrap();

    // 6. Combined patch+freq embed
    let (py_combined, _) = load_f32(&int2_st, "combined");
    let rust_combined_t = model.patch_embed.forward(input_t.clone())
        + model.freq_embed.forward(input_t.clone());
    let rust_combined: Vec<f32> = rust_combined_t.clone().squeeze::<2>().into_data().to_vec::<f32>().unwrap();
    compare("combined (pe+fe)", &rust_combined, &py_combined);

    // 7. x_tokenized (after rearrange + chan_loc_emb addition)
    let (py_xtok, _) = load_f32(&int2_st, "x_tokenized");
    let n_patches = n_t / 40;
    // Rearrange: [1, C*S, D] → [S, C, D]
    let rust_xtok_pre = rust_combined_t
        .reshape([1, n_ch, n_patches, 64])
        .swap_dims(1, 2)
        .reshape([n_patches, n_ch, 64]);
    // Chan loc emb repeat
    let rust_chan_rep = rust_cle.repeat_dim(0, n_patches);
    let rust_xtok = rust_xtok_pre + rust_chan_rep;
    let rust_xtok_vec: Vec<f32> = rust_xtok.clone().into_data().to_vec::<f32>().unwrap();
    compare("x_tokenized", &rust_xtok_vec, &py_xtok);

    // 8. Cross-attention output
    let (py_xattn, _) = load_f32(&int2_st, "cross_attn_out");
    let (rust_xattn, rust_attn_scores) = model.cross_attn.forward(rust_xtok);
    let rust_xattn_vec: Vec<f32> = rust_xattn.into_data().to_vec::<f32>().unwrap();
    compare("cross_attn_out", &rust_xattn_vec, &py_xattn);

    let (py_attn_scores, _) = load_f32(&int2_st, "cross_attn_scores");
    let rust_attn_vec: Vec<f32> = rust_attn_scores.into_data().to_vec::<f32>().unwrap();
    compare("cross_attn_scores", &rust_attn_vec, &py_attn_scores);
}
