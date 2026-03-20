/// Example 5: Embed — high-level API for producing latent embeddings.
///
/// Demonstrates:
///   - Using the LunaEncoder high-level API
///   - Channel-wise normalization
///   - Extracting and saving latent representations
///   - Statistics on the latent space
///
/// Usage:
///   cargo run --example embed --release --features hf-download
///   cargo run --example embed --release --features hf-download -- --variant large -v

#[path = "common/mod.rs"]
mod common;

use std::path::Path;
use std::time::Instant;

use burn::prelude::Backend;
use clap::Parser;
use luna_rs::{LunaEncoder, data};

#[derive(Parser, Debug)]
#[command(about = "LUNA — latent embedding extraction")]
struct Args {
    #[arg(long, default_value = "base")]
    variant: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, default_value = common::DEFAULT_REPO)]
    repo: String,
    #[arg(long, default_value = "data/embeddings.safetensors")]
    output: String,
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    dispatch_backend!(run, args)
}

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()> {
    let t0 = Instant::now();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LUNA — Latent Embedding Extraction                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 1. Resolve weights and create config
    let weights_path = common::resolve_weights(
        &args.repo, &args.variant, args.weights.as_deref(), None,
    )?;
    let cfg = common::config_for_variant(&args.variant);

    // Write a temporary config.json for LunaEncoder::load
    let config_json = serde_json::json!({
        "model": {
            "patch_size": cfg.patch_size,
            "num_queries": cfg.num_queries,
            "embed_dim": cfg.embed_dim,
            "depth": cfg.depth,
            "num_heads": cfg.num_heads,
            "mlp_ratio": cfg.mlp_ratio,
            "num_classes": cfg.num_classes,
            "drop_path": cfg.drop_path,
            "norm_eps": cfg.norm_eps,
        }
    });
    let config_path = std::env::temp_dir().join("luna_config.json");
    std::fs::write(&config_path, config_json.to_string())?;

    // 2. Load via high-level API
    println!("▸ Loading LUNA-{} via LunaEncoder API …", args.variant);
    let (encoder, ms_load) = LunaEncoder::<B>::load(
        &config_path,
        &weights_path,
        device.clone(),
    )?;
    println!("  {}  ({ms_load:.0} ms)\n", encoder.describe());

    // 3. Generate multiple "epochs" of synthetic EEG
    let n_channels = common::TUEG_CHANNELS.len();
    let n_samples = 1280;
    let n_epochs = 3;

    println!("▸ Generating {} synthetic EEG epochs ({} ch × {} samples each)\n", n_epochs, n_channels, n_samples);

    let mut all_outputs: Vec<luna_rs::EpochEmbedding> = Vec::new();

    for epoch_idx in 0..n_epochs {
        let signal = common::generate_synthetic_eeg(n_channels, n_samples, 256.0 + epoch_idx as f32);
        let batch = data::build_batch_named::<B>(signal, common::TUEG_CHANNELS, n_samples, &device);

        let t = Instant::now();
        let result = encoder.run_batch(&batch)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        // Stats
        let vals = &result.output;
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        let std: f64 = (vals.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / vals.len() as f64).sqrt();
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  Epoch {epoch_idx}: shape={:?}  mean={mean:+.4}  std={std:.4}  [{min:+.3}, {max:+.3}]  {ms:.1}ms",
            result.shape);

        if args.verbose && epoch_idx == 0 {
            // Per-channel output analysis for first epoch
            println!("\n    First 5 output values: {:?}", &vals[..5.min(vals.len())]);
        }

        all_outputs.push(result);
    }

    // 4. Save all epochs
    let encoding = luna_rs::EncodingResult {
        epochs: all_outputs,
        fif_info: None,
        ms_preproc: 0.0,
        ms_encode: t0.elapsed().as_secs_f64() * 1000.0,
    };

    if let Some(p) = Path::new(&args.output).parent() { std::fs::create_dir_all(p)?; }
    encoding.save_safetensors(&args.output)?;
    println!("\n▸ Saved {} epochs → {}", n_epochs, args.output);

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Total: {ms_total:.0} ms");
    Ok(())
}
