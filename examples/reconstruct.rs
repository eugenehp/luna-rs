/// Example 2: Reconstruct — run the full pre-training forward pass (masked reconstruction).
///
/// Demonstrates:
///   - Building an InputBatch from synthetic EEG
///   - Running the LUNA encoder in reconstruction mode (num_classes=0)
///   - Inspecting the reconstructed signal vs original
///   - Saving output to safetensors
///
/// Usage:
///   cargo run --example reconstruct --release --features hf-download
///   cargo run --example reconstruct --release --features hf-download -- --variant large -v

#[path = "common/mod.rs"]
mod common;

use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use clap::Parser;
use luna_rs::data;

#[derive(Parser, Debug)]
#[command(about = "LUNA — masked signal reconstruction")]
struct Args {
    #[arg(long, default_value = "base")]
    variant: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, default_value = common::DEFAULT_REPO)]
    repo: String,
    #[arg(long, default_value = "data/reconstructed.safetensors")]
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
    println!("║  LUNA — Reconstruction Example                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 1. Resolve and load model
    let weights_path = common::resolve_weights(
        &args.repo, &args.variant, args.weights.as_deref(), None,
    )?;
    let cfg = common::config_for_variant(&args.variant);

    println!("▸ Loading LUNA-{} (reconstruction mode) …", args.variant);
    let t_load = Instant::now();
    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), 90, &device,
    )?;
    let ms_load = t_load.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {ms_load:.0} ms\n");

    // 2. Generate synthetic EEG (22 channels, 5s @ 256Hz)
    let n_channels = common::TUEG_CHANNELS.len();
    let n_samples = 1280;
    let signal = common::generate_synthetic_eeg(n_channels, n_samples, 256.0);
    println!("▸ Input: {} channels × {} samples (5s @ 256Hz)", n_channels, n_samples);
    println!("  Channels: {}\n", common::TUEG_CHANNELS.join(", "));

    // 3. Build batch with channel name indices (enables channel identity embedding)
    let batch = data::build_batch_named::<B>(
        signal.clone(), common::TUEG_CHANNELS, n_samples, &device,
    );

    // 4. Run reconstruction
    use luna_rs::model::luna::LunaOutput;
    use luna_rs::model::rope::RotaryEmbedding;

    let head_dim = cfg.hidden_dim() / cfg.total_heads();
    let rope = RotaryEmbedding::new(head_dim, 1024, 10_000.0, &device);

    // Channel-wise normalize (matching Python inference)
    let signal_tensor = data::channel_wise_normalize(batch.signal.clone());

    println!("▸ Running forward pass …");
    let t_fwd = Instant::now();
    let output = model.forward(
        signal_tensor,
        batch.channel_locations.clone(),
        None,   // no mask at inference
        None,   // no channel names
        &rope,
    );
    let ms_fwd = t_fwd.elapsed().as_secs_f64() * 1000.0;
    println!("  Forward pass: {ms_fwd:.1} ms\n");

    match output {
        LunaOutput::Reconstruction { x_reconstructed, x_original, attention_scores } => {
            let recon_shape = x_reconstructed.dims();
            let orig_shape = x_original.dims();
            let attn_shape = attention_scores.dims();

            println!("▸ Outputs:");
            println!("  x_reconstructed: {:?}", recon_shape);
            println!("  x_original:      {:?}", orig_shape);
            println!("  attention_scores: {:?} (cross-attention Q→C affinities)\n", attn_shape);

            // Compare reconstruction to original
            let recon_vec: Vec<f32> = x_reconstructed.clone()
                .squeeze::<2>().into_data().to_vec::<f32>().unwrap();
            let orig_vec: Vec<f32> = x_original
                .squeeze::<2>().into_data().to_vec::<f32>().unwrap();

            let mse: f64 = recon_vec.iter().zip(orig_vec.iter())
                .map(|(r, o)| (*r as f64 - *o as f64).powi(2))
                .sum::<f64>() / recon_vec.len() as f64;

            println!("▸ Reconstruction quality (no masking — model sees full signal):");
            println!("  MSE: {mse:.6}");
            println!("  RMSE: {:.6}", mse.sqrt());

            if args.verbose {
                // Per-channel stats
                println!("\n  Per-channel RMSE:");
                for ch in 0..n_channels.min(8) {
                    let ch_mse: f64 = (0..n_samples)
                        .map(|t| {
                            let r = recon_vec[ch * n_samples + t] as f64;
                            let o = orig_vec[ch * n_samples + t] as f64;
                            (r - o).powi(2)
                        })
                        .sum::<f64>() / n_samples as f64;
                    println!("    {:<8} RMSE: {:.6}", common::TUEG_CHANNELS[ch], ch_mse.sqrt());
                }

                // Attention scores analysis
                let attn_vec: Vec<f32> = attention_scores
                    .into_data().to_vec::<f32>().unwrap();
                let q = cfg.num_queries;
                let c = n_channels;
                // Show which channels each query attends to most (first time patch)
                println!("\n  Query → Channel attention (first time patch, {} queries × {} channels):", q, c);
                for qi in 0..q {
                    let mut scores: Vec<(usize, f32)> = (0..c)
                        .map(|ci| (ci, attn_vec[qi * c + ci]))
                        .collect();
                    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top3: Vec<String> = scores.iter().take(3)
                        .map(|(ci, s)| format!("{}({:.3})", common::TUEG_CHANNELS[*ci], s))
                        .collect();
                    println!("    Q{qi}: top-3 = {}", top3.join(", "));
                }
            }

            // Save
            let result = luna_rs::EncodingResult {
                epochs: vec![luna_rs::EpochEmbedding {
                    output: recon_vec,
                    shape: recon_shape[1..].to_vec(),
                    chan_pos: common::tueg_positions(),
                    n_channels,
                }],
                fif_info: None,
                ms_preproc: 0.0,
                ms_encode: ms_fwd,
            };
            if let Some(p) = Path::new(&args.output).parent() { std::fs::create_dir_all(p)?; }
            result.save_safetensors(&args.output)?;
            println!("\n▸ Saved → {}", args.output);
        }
        _ => println!("  Unexpected output type (expected reconstruction)"),
    }

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Total: {ms_total:.0} ms");
    Ok(())
}
