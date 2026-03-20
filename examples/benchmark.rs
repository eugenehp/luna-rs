/// Example 4: Benchmark — measure inference latency across model variants and input sizes.
///
/// Demonstrates:
///   - Loading all three LUNA variants (base/large/huge)
///   - Timing breakdown: weight loading, forward pass
///   - Scaling behavior with channel count and sequence length
///   - LUNA's O(Q×C) channel-linear complexity
///
/// Usage:
///   cargo run --example benchmark --release --features hf-download
///   cargo run --example benchmark --release --features hf-download -- --variants base,large
///   cargo run --example benchmark --release --features hf-download -- --warmup 3 --runs 5

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use burn::prelude::*;
use clap::Parser;
use luna_rs::data;

#[derive(Parser, Debug)]
#[command(about = "LUNA — inference latency benchmark")]
struct Args {
    /// Comma-separated list of variants to benchmark.
    #[arg(long, default_value = "base")]
    variants: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, default_value = common::DEFAULT_REPO)]
    repo: String,
    /// Number of warmup runs (excluded from timing).
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Number of timed runs.
    #[arg(long, default_value_t = 5)]
    runs: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run::<burn::backend::NdArray>(burn::backend::ndarray::NdArrayDevice::Cpu, args)
}

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LUNA — Inference Benchmark                                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let variants: Vec<&str> = args.variants.split(',').map(|s| s.trim()).collect();
    let n_channels = 22;
    let n_samples = 1280;

    // Generate test signal once
    let signal = common::generate_synthetic_eeg(n_channels, n_samples, 256.0);
    let positions = common::tueg_positions();

    for variant in &variants {
        println!("━━━ LUNA-{} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", variant);

        // Load
        let weights_path = common::resolve_weights(
            &args.repo, variant, args.weights.as_deref(), None,
        )?;
        let cfg = common::config_for_variant(variant);

        let t = Instant::now();
        let model = luna_rs::weights::load_model::<B>(
            &cfg, weights_path.to_str().unwrap(), 90, &device,
        )?;
        let ms_load = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Load:    {ms_load:.0} ms");
        println!("  Config:  D={}, Q={}, depth={}, heads={}", cfg.embed_dim, cfg.num_queries, cfg.depth, cfg.num_heads);

        let head_dim = cfg.hidden_dim() / cfg.total_heads();
        let rope = luna_rs::model::rope::RotaryEmbedding::new(head_dim, 1024, 10_000.0, &device);

        // Benchmark standard input (22ch × 1280)
        println!("\n  ▸ Standard input: {}ch × {} samples", n_channels, n_samples);
        let batch = data::build_batch::<B>(
            signal.clone(), positions.clone(), None, n_channels, n_samples, &device,
        );
        let signal_norm = data::channel_wise_normalize(batch.signal.clone());

        // Warmup
        for _ in 0..args.warmup {
            let _ = model.forward(signal_norm.clone(), batch.channel_locations.clone(), None, None, &rope);
        }

        // Timed runs
        let mut times = Vec::with_capacity(args.runs);
        for _ in 0..args.runs {
            let t = Instant::now();
            let _ = model.forward(signal_norm.clone(), batch.channel_locations.clone(), None, None, &rope);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_ms = times.iter().sum::<f64>() / times.len() as f64;
        let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ms = times.iter().cloned().fold(0.0f64, f64::max);
        println!("    mean={mean_ms:.1}ms  min={min_ms:.1}ms  max={max_ms:.1}ms  (n={})", args.runs);

        // Channel scaling benchmark
        println!("\n  ▸ Channel scaling (fixed T=1280):");
        println!("    {:>6}  {:>10}", "Chans", "Mean (ms)");
        for &nc in &[4, 8, 16, 22, 32] {
            let sig = common::generate_synthetic_eeg(nc, n_samples, 256.0);
            let pos: Vec<f32> = (0..nc).flat_map(|i| {
                let frac = i as f32 / nc as f32;
                [frac * 0.1 - 0.05, frac * 0.08 - 0.04, 0.06].to_vec()
            }).collect();
            let b = data::build_batch::<B>(sig, pos, None, nc, n_samples, &device);
            let sn = data::channel_wise_normalize(b.signal.clone());
            // warmup
            let _ = model.forward(sn.clone(), b.channel_locations.clone(), None, None, &rope);
            let mut t_vec = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                let _ = model.forward(sn.clone(), b.channel_locations.clone(), None, None, &rope);
                t_vec.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            let avg = t_vec.iter().sum::<f64>() / t_vec.len() as f64;
            println!("    {:>6}  {:>7.1} ms", nc, avg);
        }
        println!();
    }

    println!("Done.");
    Ok(())
}
