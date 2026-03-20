/// Example: INT8 Quantization — quantize LUNA weights and compare accuracy.
///
/// Demonstrates:
///   - Post-training INT8 symmetric quantization
///   - Size reduction (~4× compression)
///   - Quantization error analysis
///   - Inference comparison: f32 vs quantized
///
/// Usage:
///   cargo run --example quantize --release --features hf-download
///   cargo run --example quantize --release --features hf-download -- --variant large

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use burn::prelude::*;
use clap::Parser;
use luna_rs::{data, quantize::QuantizedModel};

#[derive(Parser, Debug)]
#[command(about = "LUNA — INT8 quantization")]
struct Args {
    #[arg(long, default_value = "base")]
    variant: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, default_value = common::DEFAULT_REPO)]
    repo: String,
    /// Save quantized model to this path.
    #[arg(long, default_value = "data/luna_base_q8.bin")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run::<burn::backend::NdArray>(burn::backend::ndarray::NdArrayDevice::Cpu, args)
}

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LUNA — INT8 Post-Training Quantization                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let weights_path = common::resolve_weights(
        &args.repo, &args.variant, args.weights.as_deref(), None,
    )?;
    let cfg = common::config_for_variant(&args.variant);
    let weights_str = weights_path.to_str().unwrap();

    // 1. Quantization error analysis
    println!("▸ Analyzing quantization error on f32 weights…");
    QuantizedModel::error_stats(weights_str)?;
    println!();

    // 2. Quantize
    println!("▸ Quantizing LUNA-{} to INT8…", args.variant);
    let t = Instant::now();
    let qmodel = QuantizedModel::from_safetensors(weights_str)?;
    let ms_q = t.elapsed().as_secs_f64() * 1000.0;
    println!("  Quantized in {ms_q:.0} ms\n");

    // 3. Save
    if let Some(p) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(p)?;
    }
    qmodel.save(&args.output)?;
    let orig_size = std::fs::metadata(&weights_path)?.len();
    let q_size = std::fs::metadata(&args.output)?.len();
    println!("  Saved → {}", args.output);
    println!("  Original: {:.2} MB", orig_size as f64 / 1e6);
    println!("  Quantized: {:.2} MB", q_size as f64 / 1e6);
    println!("  Compression: {:.1}×\n", orig_size as f64 / q_size as f64);

    // 4. Load back and run inference comparison
    println!("▸ Loading quantized model and comparing inference…");
    let t = Instant::now();
    let wm = QuantizedModel::load_and_dequantize(&args.output)?;
    let ms_load = t.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded + dequantized in {ms_load:.0} ms");

    // Build the model from dequantized weights
    let model_q = luna_rs::weights::load_model_from_wm::<B>(&cfg, &wm, 90, &device)?;
    let model_f32 = luna_rs::weights::load_model::<B>(
        &cfg, weights_str, 90, &device,
    )?;

    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(
        cfg.hidden_dim() / cfg.total_heads(), 1024, 10_000.0, &device,
    );

    // Generate test input
    let n_ch = common::TUEG_CHANNELS.len();
    let n_t = 1280;
    let signal = common::generate_synthetic_eeg(n_ch, n_t, 256.0);
    let batch = data::build_batch_named::<B>(signal, common::TUEG_CHANNELS, n_t, &device);
    let sig_norm = data::channel_wise_normalize(batch.signal.clone());

    // Run both models
    let out_f32 = model_f32.forward(
        sig_norm.clone(), batch.channel_locations.clone(), None, batch.channel_names.clone(), &rope,
    );
    let out_q8 = model_q.forward(
        sig_norm, batch.channel_locations.clone(), None, batch.channel_names.clone(), &rope,
    );

    let (f32_vec, q8_vec) = match (out_f32, out_q8) {
        (luna_rs::model::luna::LunaOutput::Reconstruction { x_reconstructed: r1, .. },
         luna_rs::model::luna::LunaOutput::Reconstruction { x_reconstructed: r2, .. }) => {
            (r1.into_data().to_vec::<f32>().unwrap(),
             r2.into_data().to_vec::<f32>().unwrap())
        }
        _ => anyhow::bail!("expected reconstruction output"),
    };

    // Compare
    let n = f32_vec.len() as f64;
    let max_err: f32 = f32_vec.iter().zip(q8_vec.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let rmse: f64 = (f32_vec.iter().zip(q8_vec.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>() / n).sqrt();
    let f32_std: f64 = {
        let mean = f32_vec.iter().map(|&v| v as f64).sum::<f64>() / n;
        (f32_vec.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
    };
    let corr = {
        let f32_mean = f32_vec.iter().map(|&v| v as f64).sum::<f64>() / n;
        let q8_mean = q8_vec.iter().map(|&v| v as f64).sum::<f64>() / n;
        let cov: f64 = f32_vec.iter().zip(q8_vec.iter())
            .map(|(&a, &b)| (a as f64 - f32_mean) * (b as f64 - q8_mean)).sum::<f64>() / n;
        let q8_std = (q8_vec.iter().map(|&v| (v as f64 - q8_mean).powi(2)).sum::<f64>() / n).sqrt();
        cov / (f32_std * q8_std + 1e-10)
    };

    println!("\n▸ F32 vs INT8 inference comparison:");
    println!("  Max absolute error:  {max_err:.6}");
    println!("  RMSE:                {rmse:.6}");
    println!("  Relative RMSE:       {:.4} ({:.2}%)", rmse / f32_std, rmse / f32_std * 100.0);
    println!("  Pearson correlation:  {corr:.6}");

    Ok(())
}
