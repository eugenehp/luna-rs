//! Integration tests for LUNA forward pass with real pretrained weights.
//!
//! These tests load LUNA-Base weights from the HuggingFace cache and verify
//! output shapes, value ranges, and attention score properties.
//!
//! Skipped automatically when weights are not cached — run:
//!   cargo test --release
//! after downloading weights (e.g. via the `load_and_inspect` example).

use std::path::PathBuf;

use burn::backend::NdArray as B;
type Dev = burn::backend::ndarray::NdArrayDevice;

fn device() -> Dev { Dev::Cpu }

/// Return path to LUNA-Base weights, or None if not cached.
fn find_weights() -> Option<PathBuf> {
    let base = dirs().join("models--thorir--LUNA").join("snapshots");
    if !base.exists() { return None; }
    let mut dirs: Vec<_> = std::fs::read_dir(&base).ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    let snap = dirs.last()?.path();
    let w = snap.join("LUNA_base.safetensors");
    if w.exists() { Some(w) } else { None }
}

fn dirs() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HOME") { return PathBuf::from(v).join("hub"); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}

fn luna_base_config() -> luna_rs::ModelConfig {
    luna_rs::ModelConfig {
        patch_size: 40, num_queries: 4, embed_dim: 64,
        depth: 8, num_heads: 2, mlp_ratio: 4.0,
        num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
    }
}

// ── Test: forward pass with real weights ─────────────────────────────────────

#[test]
fn reconstruction_shapes_and_values() {
    let Some(weights_path) = find_weights() else {
        eprintln!("SKIP: LUNA-Base weights not cached. Run an example with --features hf-download first.");
        return;
    };

    let dev = device();
    let cfg = luna_base_config();

    // Load model
    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), luna_rs::VOCAB_SIZE, &dev,
    ).expect("load model");

    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(32, 1024, 10_000.0, &dev);

    // Build input with channel names
    let n_ch = luna_rs::TUEG_CHANNELS.len(); // 22
    let n_t = 1280; // 5s @ 256 Hz
    let signal: Vec<f32> = (0..n_ch * n_t)
        .map(|i| (i as f32 * 0.01).sin() * 1e-5)
        .collect();

    let batch = luna_rs::build_batch_named::<B>(
        signal, luna_rs::TUEG_CHANNELS, n_t, &dev,
    );

    // Verify batch has channel indices
    assert!(batch.channel_names.is_some(), "channel_names should be set");

    // Run forward
    let signal_norm = luna_rs::data::channel_wise_normalize(batch.signal.clone());
    let output = model.forward(
        signal_norm,
        batch.channel_locations.clone(),
        None,
        batch.channel_names.clone(),
        &rope,
    );

    match output {
        luna_rs::model::luna::LunaOutput::Reconstruction {
            x_reconstructed, x_original, attention_scores
        } => {
            // Shape checks
            assert_eq!(x_reconstructed.dims(), [1, n_ch, n_t],
                "reconstructed shape mismatch");
            assert_eq!(x_original.dims(), [1, n_ch, n_t],
                "original shape mismatch");

            let num_patches = n_t / cfg.patch_size; // 32
            assert_eq!(attention_scores.dims()[1], cfg.num_queries,
                "attention scores should have Q={} queries", cfg.num_queries);
            assert_eq!(attention_scores.dims()[2], n_ch,
                "attention scores should have C={} channels", n_ch);

            // Value checks: no NaN, no Inf, reasonable range
            let recon_vec: Vec<f32> = x_reconstructed.into_data().to_vec::<f32>().unwrap();
            assert!(!recon_vec.iter().any(|v| v.is_nan()), "reconstruction contains NaN");
            assert!(!recon_vec.iter().any(|v| v.is_infinite()), "reconstruction contains Inf");

            let mean_abs: f64 = recon_vec.iter().map(|v| v.abs() as f64).sum::<f64>()
                / recon_vec.len() as f64;
            assert!(mean_abs < 10.0, "mean |recon| = {mean_abs}, unexpectedly large");
            assert!(mean_abs > 1e-10, "mean |recon| = {mean_abs}, suspiciously small (all zeros?)");

            // Attention scores: should be valid softmax output per query
            let attn_vec: Vec<f32> = attention_scores.into_data().to_vec::<f32>().unwrap();
            assert!(!attn_vec.iter().any(|v| v.is_nan()), "attention scores contain NaN");
            assert!(attn_vec.iter().all(|&v| v >= 0.0), "attention scores should be non-negative");

            // Each query's scores across channels should sum to ~1 (softmax)
            // (averaged across heads, so may not be exactly 1)
            let q = cfg.num_queries;
            let c = n_ch;
            let first_patch_attn = &attn_vec[..q * c]; // first time patch
            for qi in 0..q {
                let sum: f32 = (0..c).map(|ci| first_patch_attn[qi * c + ci]).sum();
                assert!((sum - 1.0).abs() < 0.1,
                    "query {qi} attention sum = {sum}, expected ~1.0");
            }

            println!("✓ Reconstruction: shape=[1, {n_ch}, {n_t}], mean|out|={mean_abs:.6}");
            println!("  Attention: [{}, {q}, {c}]", num_patches * 1);
        }
        _ => panic!("Expected Reconstruction output, got Classification"),
    }
}

// ── Test: channel vocabulary integration ─────────────────────────────────────

#[test]
fn channel_vocab_indices_match_embedding_size() {
    let Some(weights_path) = find_weights() else {
        eprintln!("SKIP: weights not cached");
        return;
    };

    // Load weights and check embedding size
    let wm = luna_rs::weights::WeightMap::from_file(
        weights_path.to_str().unwrap()
    ).expect("load weight map");

    let (_emb_data, emb_shape) = wm.tensors.get("channel_emb.embeddings.weight")
        .expect("channel_emb.embeddings.weight not found");
    assert_eq!(emb_shape[0], luna_rs::VOCAB_SIZE,
        "embedding vocab size {} != VOCAB_SIZE {}", emb_shape[0], luna_rs::VOCAB_SIZE);

    // Verify all TUEG channel indices are in range
    for &ch in luna_rs::TUEG_CHANNELS {
        let idx = luna_rs::channel_index(ch).expect(&format!("channel '{}' not in vocab", ch));
        assert!(idx < luna_rs::VOCAB_SIZE, "index {} out of range for '{}'", idx, ch);
    }
}

// ── Test: forward pass without channel names (None) ──────────────────────────

#[test]
fn reconstruction_without_channel_names() {
    let Some(weights_path) = find_weights() else {
        eprintln!("SKIP: weights not cached");
        return;
    };

    let dev = device();
    let cfg = luna_base_config();
    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), luna_rs::VOCAB_SIZE, &dev,
    ).expect("load model");
    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(32, 1024, 10_000.0, &dev);

    let n_ch = 22;
    let n_t = 1280;
    let signal = vec![0.0f32; n_ch * n_t];
    let positions = vec![0.05f32; n_ch * 3];

    // Build batch WITHOUT channel names
    let batch = luna_rs::data::build_batch::<B>(
        signal, positions, None, n_ch, n_t, &dev,
    );
    assert!(batch.channel_names.is_none());

    let signal_norm = luna_rs::data::channel_wise_normalize(batch.signal.clone());
    let output = model.forward(signal_norm, batch.channel_locations.clone(), None, None, &rope);

    match output {
        luna_rs::model::luna::LunaOutput::Reconstruction { x_reconstructed, .. } => {
            assert_eq!(x_reconstructed.dims(), [1, n_ch, n_t]);
            let vals: Vec<f32> = x_reconstructed.into_data().to_vec::<f32>().unwrap();
            assert!(!vals.iter().any(|v| v.is_nan()), "NaN without channel names");
            println!("✓ Forward pass without channel_names works");
        }
        _ => panic!("Expected Reconstruction"),
    }
}

// ── Test: different channel counts ───────────────────────────────────────────

#[test]
fn variable_channel_count() {
    let Some(weights_path) = find_weights() else {
        eprintln!("SKIP: weights not cached");
        return;
    };

    let dev = device();
    let cfg = luna_base_config();
    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), luna_rs::VOCAB_SIZE, &dev,
    ).expect("load model");
    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(32, 1024, 10_000.0, &dev);

    for n_ch in [4, 8, 16, 22, 29] {
        let n_t = 1280;
        let signal = vec![0.01f32; n_ch * n_t];
        let positions = vec![0.05f32; n_ch * 3];
        let batch = luna_rs::data::build_batch::<B>(
            signal, positions, None, n_ch, n_t, &dev,
        );
        let signal_norm = luna_rs::data::channel_wise_normalize(batch.signal.clone());
        let output = model.forward(signal_norm, batch.channel_locations.clone(), None, None, &rope);

        match output {
            luna_rs::model::luna::LunaOutput::Reconstruction { x_reconstructed, .. } => {
                assert_eq!(x_reconstructed.dims(), [1, n_ch, n_t],
                    "shape mismatch for {n_ch} channels");
                let vals: Vec<f32> = x_reconstructed.into_data().to_vec::<f32>().unwrap();
                assert!(!vals.iter().any(|v| v.is_nan()), "NaN for {n_ch} channels");
            }
            _ => panic!("Expected Reconstruction for {n_ch} channels"),
        }
        println!("  ✓ {n_ch} channels OK");
    }
}
