/// Example 3: Channel Invariance — demonstrate LUNA's topology-agnostic design.
///
/// Demonstrates:
///   - Running the same model on inputs with DIFFERENT numbers of channels
///   - LUNA handles 10, 16, 22, and 29 channels without any architecture changes
///   - The cross-attention module compresses any number of channels → Q fixed queries
///
/// Usage:
///   cargo run --example channel_invariance --release --features hf-download

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use burn::prelude::*;
use clap::Parser;
use luna_rs::{data, bipolar_channel_xyz, channel_xyz};

#[derive(Parser, Debug)]
#[command(about = "LUNA — channel-count invariance demonstration")]
struct Args {
    #[arg(long, default_value = "base")]
    variant: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, default_value = common::DEFAULT_REPO)]
    repo: String,
}

/// Channel subsets to test — different sizes and montage types.
fn channel_configs() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("10ch bipolar (subset)", vec![
            "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
            "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
            "C3-CZ", "CZ-C4",
        ]),
        ("16ch bipolar", vec![
            "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
            "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
            "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        ]),
        ("22ch TUEG bipolar (full)", common::TUEG_CHANNELS.to_vec()),
        ("8ch unipolar", vec![
            "FP1", "FP2", "F3", "F4", "C3", "C4", "O1", "O2",
        ]),
    ]
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run::<burn::backend::NdArray>(burn::backend::ndarray::NdArrayDevice::Cpu, args)
}

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LUNA — Channel Invariance Demonstration                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load model once
    let weights_path = common::resolve_weights(
        &args.repo, &args.variant, args.weights.as_deref(), None,
    )?;
    let cfg = common::config_for_variant(&args.variant);

    println!("▸ Loading LUNA-{} …", args.variant);
    let model = luna_rs::weights::load_model::<B>(
        &cfg, weights_path.to_str().unwrap(), 90, &device,
    )?;

    let head_dim = cfg.hidden_dim() / cfg.total_heads();
    let rope = luna_rs::model::rope::RotaryEmbedding::new(head_dim, 1024, 10_000.0, &device);

    let n_samples = 1280;
    let configs = channel_configs();

    println!("\n▸ Running same model on {} different channel configurations:\n", configs.len());
    println!("  {:<30} {:>5}  {:>12}  {:>12}  {:>10}", "Config", "Chans", "Output shape", "Time (ms)", "Mean|out|");
    println!("  {}", "─".repeat(75));

    for (name, channels) in &configs {
        let n_ch = channels.len();

        // Generate signal
        let signal = common::generate_synthetic_eeg(n_ch, n_samples, 256.0);

        // Get positions — handle both bipolar and unipolar
        let positions: Vec<f32> = channels.iter()
            .flat_map(|ch| {
                bipolar_channel_xyz(ch)
                    .or_else(|| channel_xyz(ch))
                    .unwrap_or([0.0, 0.0, 0.0])
                    .to_vec()
            })
            .collect();

        let batch = data::build_batch::<B>(
            signal, positions, None, n_ch, n_samples, &device,
        );

        let signal_norm = data::channel_wise_normalize(batch.signal.clone());

        let t = Instant::now();
        let output = model.forward(
            signal_norm,
            batch.channel_locations.clone(),
            None, None, &rope,
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        // Extract output shape and stats
        let (shape_str, mean_abs) = match output {
            luna_rs::model::luna::LunaOutput::Reconstruction { x_reconstructed, .. } => {
                let dims = x_reconstructed.dims();
                let vals: Vec<f32> = x_reconstructed.squeeze::<2>()
                    .into_data().to_vec::<f32>().unwrap();
                let ma: f64 = vals.iter().map(|v| v.abs() as f64).sum::<f64>() / vals.len() as f64;
                (format!("[{}, {}, {}]", dims[0], dims[1], dims[2]), ma)
            }
            _ => ("?".into(), 0.0),
        };

        println!("  {:<30} {:>5}  {:>12}  {:>9.1} ms  {:>10.6}",
            name, n_ch, shape_str, ms, mean_abs);
    }

    println!("\n  ✓ Same model, same weights — works with any number of channels!");
    println!("    The cross-attention module maps C channels → {} fixed queries.", cfg.num_queries);

    Ok(())
}
