/// Example 1: Load & Inspect — download weights, load model, print architecture info.
///
/// Demonstrates:
///   - Automatic weight download from HuggingFace (thorir/LUNA)
///   - Model config inference from variant name
///   - Weight loading and tensor key inspection
///   - Model architecture summary
///
/// Usage:
///   cargo run --example load_and_inspect --release --features hf-download
///   cargo run --example load_and_inspect --release --features hf-download -- --variant large

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;
use clap::Parser;
use luna_rs::weights::WeightMap;

#[derive(Parser, Debug)]
#[command(about = "LUNA — load model and inspect architecture")]
struct Args {
    /// Model variant: base, large, or huge.
    #[arg(long, default_value = "base")]
    variant: String,

    /// Explicit weights path (skip HF download).
    #[arg(long)]
    weights: Option<String>,

    /// HuggingFace repo ID.
    #[arg(long, default_value = common::DEFAULT_REPO)]
    repo: String,

    /// Print all weight tensor keys and shapes.
    #[arg(long)]
    print_keys: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LUNA — Load & Inspect                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 1. Resolve weights
    println!("▸ Resolving weights for LUNA-{} …", args.variant);
    let weights_path = common::resolve_weights(
        &args.repo, &args.variant, args.weights.as_deref(), None,
    )?;
    println!("  Weights: {}\n", weights_path.display());

    // 2. Load and inspect weight map
    let wm = WeightMap::from_file(weights_path.to_str().unwrap())?;
    println!("▸ Weight file contains {} tensors", wm.tensors.len());

    if args.print_keys {
        println!("\n  All weight keys:");
        wm.print_keys();
        println!();
    }

    // Show key statistics
    let total_params: usize = wm.tensors.values().map(|(d, _)| d.len()).sum();
    let total_bytes = total_params * 4;  // f32
    println!("  Total parameters: {:.2}M ({:.1} MB as f32)",
        total_params as f64 / 1e6, total_bytes as f64 / 1e6);

    // Count by component
    let components = ["patch_embed", "freq_embed", "channel_location", "cross_attn",
                      "blocks", "norm", "decoder_head", "channel_emb", "mask_token"];
    println!("\n  Parameter breakdown:");
    for comp in &components {
        let count: usize = wm.tensors.iter()
            .filter(|(k, _)| k.starts_with(comp))
            .map(|(_, (d, _))| d.len())
            .sum();
        if count > 0 {
            println!("    {:<30} {:>8.2}M", comp, count as f64 / 1e6);
        }
    }

    // 3. Load model
    println!("\n▸ Loading LUNA-{} model …", args.variant);
    let cfg = common::config_for_variant(&args.variant);
    println!("  Config: embed_dim={}, queries={}, depth={}, heads={}, patch_size={}",
        cfg.embed_dim, cfg.num_queries, cfg.depth, cfg.num_heads, cfg.patch_size);
    println!("  Hidden dim (Q×D): {}", cfg.hidden_dim());
    println!("  FFN hidden dim:   {}", cfg.ffn_hidden_dim());
    println!("  Total attn heads: {}", cfg.total_heads());

    let t_load = Instant::now();
    let _model = luna_rs::weights::load_model::<burn::backend::NdArray>(
        &cfg, weights_path.to_str().unwrap(), 90, &burn::backend::ndarray::NdArrayDevice::Cpu,
    )?;
    let ms_load = t_load.elapsed().as_secs_f64() * 1000.0;
    println!("  Model loaded in {ms_load:.0} ms");

    println!("\n▸ Architecture:");
    println!("  Input:  [B, C, T]  where C=variable, T=1280 (5s @ 256Hz)");
    println!("  1. PatchEmbed:    [B, C, T] → [B, C×S, D]       S=T/{}, D={}", cfg.patch_size, cfg.embed_dim);
    println!("  2. FreqEmbed:     [B, C, T] → [B, C×S, D]       (added to PatchEmbed)");
    println!("  3. ChanLocEmbed:  [B, C, 3] → [B, C, D]         (NeRF + MLP)");
    println!("  4. Rearrange:     [B, C×S, D] → [B×S, C, D]     (per-patch channel grouping)");
    println!("  5. CrossAttn:     [B×S, C, D] → [B×S, Q, D]     Q={} queries", cfg.num_queries);
    println!("  6. Temporal:      [B, S, Q×D] → [B, S, Q×D]     {} × RotaryTransformerBlock", cfg.depth);
    println!("  7. Head:          Reconstruction or Classification");

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Total time: {ms_total:.0} ms");
    Ok(())
}
