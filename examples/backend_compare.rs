//! LUNA encoder benchmark across every compiled-in Burn / RLX backend.
//!
//! ```sh
//! # CPU quick compare:
//! cargo run --example backend_compare --release \
//!     --no-default-features --features burn,rlx,ndarray,rlx-cpu
//!
//! # Apple Silicon (Burn wgpu + RLX metal/mlx):
//! cargo run --example backend_compare --release \
//!     --no-default-features \
//!     --features burn,rlx,ndarray,wgpu,rlx-cpu,rlx-metal,rlx-mlx
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "LUNA — Burn vs RLX encoder benchmark (all compiled backends)")]
struct Args {
    /// Comma-separated channel counts (default: 22 = TUEG).
    #[arg(long, default_value = "22")]
    channels: String,

    /// Time samples per epoch (5 s @ 256 Hz).
    #[arg(long, default_value_t = 1280)]
    time_samples: usize,

    #[arg(long, default_value_t = 3)]
    runs: usize,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    #[arg(long, env = "LUNA_WEIGHTS")]
    weights: Option<String>,

    #[arg(long, env = "LUNA_CONFIG")]
    config: Option<String>,

    /// Report max_abs vs Burn/NdArray CPU for each RLX backend.
    #[arg(long)]
    parity: bool,
}

struct BenchResult {
    engine: String,
    backend: String,
    n_channels: usize,
    n_samples: usize,
    runs: Vec<f64>,
    /// Set when `--parity` and Burn CPU reference ran.
    parity_max_abs: Option<f32>,
}

impl BenchResult {
    fn label(&self) -> String {
        format!("{}/{}", self.engine, self.backend)
    }
    fn min_ms(&self) -> f64 {
        self.runs.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn mean_ms(&self) -> f64 {
        self.runs.iter().sum::<f64>() / self.runs.len() as f64
    }
}

fn resolve_paths(args: &Args) -> anyhow::Result<(PathBuf, PathBuf)> {
    let config = args
        .config
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/config.json")
        });
    if !config.exists() {
        let cfg = serde_json::json!({
            "model": {
                "patch_size": 40, "num_queries": 4, "embed_dim": 64,
                "depth": 8, "num_heads": 2, "mlp_ratio": 4.0,
                "num_classes": 0, "norm_eps": 1e-5
            }
        });
        std::fs::write(&config, serde_json::to_string_pretty(&cfg)?)?;
    }

    let weights = if let Some(w) = &args.weights {
        PathBuf::from(w)
    } else if let Ok(w) = std::env::var("LUNA_WEIGHTS") {
        PathBuf::from(w)
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let snaps = PathBuf::from(home)
            .join(".cache/huggingface/hub/models--thorir--LUNA/snapshots");
        let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .collect();
        dirs.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        dirs.last()
            .map(|d| d.path().join("LUNA_base.safetensors"))
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/LUNA_base.safetensors")
            })
    };

    if !weights.exists() {
        anyhow::bail!(
            "weights not found at {} — set LUNA_WEIGHTS or run download_weights",
            weights.display()
        );
    }
    Ok((weights, config))
}

fn synthetic_epoch(n_ch: usize, n_t: usize) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let signal: Vec<f32> = (0..n_ch * n_t)
        .map(|i| (i as f32 * 0.01).sin() * 1e-5)
        .collect();
    let chan_pos = vec![0.01f32; n_ch * 3];
    let chan_idx: Vec<i32> = (0..n_ch as i32).collect();
    (signal, chan_pos, chan_idx)
}

// ── Burn ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
mod burn_bench {
    use super::*;
    use burn::prelude::*;
    use luna_rs::data::build_batch;
    use luna_rs::LunaEncoder;

    fn bench_encoder<B: Backend>(
        engine: &str,
        backend: &str,
        enc: &LunaEncoder<B>,
        device: &B::Device,
        channel_counts: &[usize],
        n_t: usize,
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        let mut out = Vec::new();
        for &n_ch in channel_counts {
            let (signal, pos, idx) = synthetic_epoch(n_ch, n_t);
            let names: Vec<i64> = idx.iter().map(|&v| v as i64).collect();
            for _ in 0..n_warmup {
                let batch = build_batch::<B>(
                    signal.clone(),
                    pos.clone(),
                    Some(names.clone()),
                    n_ch,
                    n_t,
                    &device,
                );
                let _ = enc.run_batch(&batch);
            }
            let mut runs = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                let batch = build_batch::<B>(
                    signal.clone(),
                    pos.clone(),
                    Some(names.clone()),
                    n_ch,
                    n_t,
                    &device,
                );
                let t = Instant::now();
                let _ = enc.run_batch(&batch);
                runs.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            out.push(BenchResult {
                engine: engine.into(),
                backend: backend.into(),
                n_channels: n_ch,
                n_samples: n_t,
                runs,
                parity_max_abs: None,
            });
        }
        out
    }

    pub fn run_all(
        config_path: &Path,
        weights_path: &Path,
        channel_counts: &[usize],
        n_t: usize,
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        let mut all = Vec::new();

        #[cfg(feature = "ndarray")]
        {
            use burn::backend::{ndarray::NdArrayDevice, NdArray};
            let name = if cfg!(feature = "blas-accelerate") {
                "NdArray+Accelerate"
            } else if cfg!(feature = "openblas-system") {
                "NdArray+OpenBLAS"
            } else {
                "NdArray"
            };
            eprint!("  Burn/{name:<20} ");
            let dev = NdArrayDevice::Cpu;
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                LunaEncoder::<NdArray>::load(config_path, weights_path, dev.clone())
            })) {
                Ok(Ok((enc, _))) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    bench_encoder("Burn", name, &enc, &dev, channel_counts, n_t, n_warmup, n_runs)
                })) {
                    Ok(r) => {
                        all.extend(r);
                        eprintln!("ok");
                    }
                    Err(_) => eprintln!("SKIP (panic during run)"),
                },
                Ok(Err(e)) => eprintln!("SKIP ({e})"),
                Err(_) => eprintln!("SKIP (panic on load)"),
            }
        }

        #[cfg(feature = "wgpu")]
        {
            use burn::backend::{wgpu::WgpuDevice, Wgpu};
            eprint!("  Burn/{:<20} ", "wgpu f32");
            let dev = WgpuDevice::DefaultDevice;
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                LunaEncoder::<Wgpu>::load(config_path, weights_path, dev.clone())
            })) {
                Ok(Ok((enc, _))) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    bench_encoder("Burn", "wgpu f32", &enc, &dev, channel_counts, n_t, n_warmup, n_runs)
                })) {
                    Ok(r) => {
                        all.extend(r);
                        eprintln!("ok");
                    }
                    Err(_) => eprintln!("SKIP (panic during run)"),
                },
                Ok(Err(e)) => eprintln!("SKIP ({e})"),
                Err(_) => eprintln!("SKIP (panic on load)"),
            }
        }

        all
    }

    /// Burn/NdArray reference outputs for parity checks.
    pub fn reference_outputs(
        config_path: &Path,
        weights_path: &Path,
        channel_counts: &[usize],
        n_t: usize,
    ) -> anyhow::Result<std::collections::HashMap<usize, Vec<f32>>> {
        use burn::backend::{ndarray::NdArrayDevice, NdArray};
        use luna_rs::model::luna::LunaOutput;

        let cfg_str = std::fs::read_to_string(config_path)?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: luna_rs::ModelConfig = serde_json::from_value(
            hf_val.get("model").cloned().unwrap_or(hf_val),
        )?;
        let device = NdArrayDevice::Cpu;
        let model = luna_rs::weights::load_model::<NdArray>(
            &model_cfg,
            weights_path.to_str().unwrap(),
            luna_rs::VOCAB_SIZE,
            &device,
        )?;
        let rope = luna_rs::model::rope::RotaryEmbedding::<NdArray>::new(
            model_cfg.head_dim(),
            1024,
            10_000.0,
            &device,
        );
        let mut refs = std::collections::HashMap::new();
        for &n_ch in channel_counts {
            let (signal, pos, idx) = synthetic_epoch(n_ch, n_t);
            let names: Vec<i64> = idx.iter().map(|&v| v as i64).collect();
            let batch = build_batch::<NdArray>(signal, pos, Some(names), n_ch, n_t, &device);
            let signal_norm = luna_rs::data::channel_wise_normalize(batch.signal.clone());
            let out = model.forward(
                signal_norm,
                batch.channel_locations.clone(),
                None,
                batch.channel_names.clone(),
                &rope,
            );
            let vec = match out {
                LunaOutput::Reconstruction { x_reconstructed, .. } => x_reconstructed
                    .squeeze::<2>()
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?,
                LunaOutput::Classification { logits, .. } => logits
                    .squeeze::<1>()
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?,
            };
            refs.insert(n_ch, vec);
        }
        Ok(refs)
    }
}

// ── RLX ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "rlx")]
mod rlx_bench {
    use super::*;
    use luna_rs::rlx::{LunaEncoder, RunEpochOpts};

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn bench_encoder(
        backend: &str,
        enc: &mut LunaEncoder,
        channel_counts: &[usize],
        n_t: usize,
        n_warmup: usize,
        n_runs: usize,
        refs: Option<&std::collections::HashMap<usize, Vec<f32>>>,
    ) -> Vec<BenchResult> {
        let mut out = Vec::new();
        for &n_ch in channel_counts {
            for _ in 0..n_warmup {
                let (s, p, i) = synthetic_epoch(n_ch, n_t);
                let _ = enc
                    .run_epoch_opts(&s, &p, Some(&i), n_ch, n_t, RunEpochOpts::default())
                    .expect("warmup");
            }
            let mut runs = Vec::with_capacity(n_runs);
            let mut last_out: Option<Vec<f32>> = None;
            for _ in 0..n_runs {
                let (s, p, i) = synthetic_epoch(n_ch, n_t);
                let t0 = Instant::now();
                let ep = enc
                    .run_epoch_opts(&s, &p, Some(&i), n_ch, n_t, RunEpochOpts::default())
                    .expect("run_epoch");
                runs.push(t0.elapsed().as_secs_f64() * 1000.0);
                last_out = Some(ep.output);
            }
            let parity_max_abs = refs
                .and_then(|r| r.get(&n_ch))
                .zip(last_out.as_ref())
                .map(|(refv, got)| max_abs(got, refv));
            out.push(BenchResult {
                engine: "RLX".into(),
                backend: backend.into(),
                n_channels: n_ch,
                n_samples: n_t,
                runs,
                parity_max_abs,
            });
        }
        out
    }

    struct RlxBenchCtx<'a> {
        label: &'a str,
        device: rlx::Device,
        config_path: &'a Path,
        weights_path: &'a Path,
        channel_counts: &'a [usize],
        n_t: usize,
        n_warmup: usize,
        n_runs: usize,
        refs: Option<&'a std::collections::HashMap<usize, Vec<f32>>>,
    }

    fn try_bench(ctx: RlxBenchCtx<'_>) -> Vec<BenchResult> {
        eprint!("  RLX/{:<20} ", ctx.label);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            LunaEncoder::load(ctx.config_path, ctx.weights_path, ctx.device)
        })) {
            Ok(Ok((mut enc, _))) => {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    bench_encoder(
                        ctx.label,
                        &mut enc,
                        ctx.channel_counts,
                        ctx.n_t,
                        ctx.n_warmup,
                        ctx.n_runs,
                        ctx.refs,
                    )
                }));
                match r {
                    Ok(v) => {
                        eprintln!("ok");
                        v
                    }
                    Err(_) => {
                        eprintln!("SKIP (panic during run)");
                        Vec::new()
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("SKIP ({e})");
                Vec::new()
            }
            Err(_) => {
                eprintln!("SKIP (panic on load)");
                Vec::new()
            }
        }
    }

    pub fn run_all(
        config_path: &Path,
        weights_path: &Path,
        channel_counts: &[usize],
        n_t: usize,
        n_warmup: usize,
        n_runs: usize,
        refs: Option<&std::collections::HashMap<usize, Vec<f32>>>,
    ) -> Vec<BenchResult> {
        let mut all = Vec::new();
        let base = |label: &'static str, device: rlx::Device| RlxBenchCtx {
            label,
            device,
            config_path,
            weights_path,
            channel_counts,
            n_t,
            n_warmup,
            n_runs,
            refs,
        };

        #[cfg(feature = "rlx-cpu")]
        {
            let name: &'static str = if cfg!(feature = "rlx-blas-accelerate") {
                "CPU+Accelerate"
            } else if cfg!(feature = "rlx-blas-openblas") {
                "CPU+OpenBLAS"
            } else {
                "CPU"
            };
            all.extend(try_bench(base(name, rlx::Device::Cpu)));
        }

        #[cfg(feature = "rlx-metal")]
        all.extend(try_bench(base("Metal", rlx::Device::Metal)));

        #[cfg(feature = "rlx-mlx")]
        all.extend(try_bench(base("MLX", rlx::Device::Mlx)));

        #[cfg(feature = "rlx-gpu")]
        all.extend(try_bench(base("wgpu", rlx::Device::Gpu)));

        #[cfg(feature = "rlx-cuda")]
        all.extend(try_bench(base("CUDA", rlx::Device::Cuda)));

        #[cfg(feature = "rlx-rocm")]
        all.extend(try_bench(base("ROCm", rlx::Device::Rocm)));

        #[cfg(feature = "rlx-tpu")]
        all.extend(try_bench(base("TPU", rlx::Device::Tpu)));

        all
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _threads = luna_rs::init_threads(args.threads);

    let channel_counts: Vec<usize> = args
        .channels
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    let (weights_path, config_path) = resolve_paths(&args)?;

    println!("=== LUNA Burn vs RLX — backend benchmark ===");
    println!("  weights  : {}", weights_path.display());
    println!("  config   : {}", config_path.display());
    println!("  channels : {:?}", channel_counts);
    println!("  T        : {}", args.time_samples);
    println!("  runs     : {} (+ {} warmup)", args.runs, args.warmup);
    if args.parity {
        println!("  parity   : vs Burn/NdArray CPU");
    }
    println!();

    #[cfg(all(feature = "burn", feature = "ndarray"))]
    let refs = if args.parity {
        println!("Building Burn/NdArray reference outputs…");
        let r = burn_bench::reference_outputs(
            &config_path,
            &weights_path,
            &channel_counts,
            args.time_samples,
        )?;
        println!();
        Some(r)
    } else {
        None
    };

    #[cfg(not(all(feature = "burn", feature = "ndarray")))]
    let refs: Option<std::collections::HashMap<usize, Vec<f32>>> = None;
    if args.parity && refs.is_none() {
        eprintln!("Note: --parity needs `--features burn,ndarray`");
    }

    let mut results = Vec::new();

    #[cfg(feature = "burn")]
    {
        println!("Burn backends:");
        results.extend(burn_bench::run_all(
            &config_path,
            &weights_path,
            &channel_counts,
            args.time_samples,
            args.warmup,
            args.runs,
        ));
        println!();
    }

    #[cfg(feature = "rlx")]
    {
        println!("RLX backends:");
        results.extend(rlx_bench::run_all(
            &config_path,
            &weights_path,
            &channel_counts,
            args.time_samples,
            args.warmup,
            args.runs,
            refs.as_ref(),
        ));
        println!();
    }

    if results.is_empty() {
        anyhow::bail!(
            "no backends ran — enable `rlx` (+ `rlx-cpu`, `rlx-metal`, …) and/or `burn` (+ `ndarray`, …)"
        );
    }

    let show_parity = args.parity && results.iter().any(|r| r.parity_max_abs.is_some());
    if show_parity {
        println!(
            "{:<26} {:>4} {:>6} {:>10} {:>10} {:>12}",
            "Engine/Backend", "Ch", "T", "Min(ms)", "Mean(ms)", "Parity"
        );
        println!("{}", "─".repeat(76));
        for r in &results {
            let p = r
                .parity_max_abs
                .map(|v| format!("{v:.2e}"))
                .unwrap_or_else(|| "—".into());
            println!(
                "{:<26} {:>4} {:>6} {:>10.1} {:>10.1} {:>12}",
                r.label(),
                r.n_channels,
                r.n_samples,
                r.min_ms(),
                r.mean_ms(),
                p,
            );
        }
    } else {
        println!(
            "{:<26} {:>4} {:>6} {:>10} {:>10}",
            "Engine/Backend", "Ch", "T", "Min(ms)", "Mean(ms)"
        );
        println!("{}", "─".repeat(62));
        for r in &results {
            println!(
                "{:<26} {:>4} {:>6} {:>10.1} {:>10.1}",
                r.label(),
                r.n_channels,
                r.n_samples,
                r.min_ms(),
                r.mean_ms(),
            );
        }
    }

    #[cfg(all(feature = "burn", feature = "rlx"))]
    {
        println!();
        println!("── RLX best / Burn best (speedup > 1 ⇒ RLX faster) ──");
        for &ch in &channel_counts {
            let burn_best = results
                .iter()
                .filter(|r| r.engine == "Burn" && r.n_channels == ch)
                .map(|r| r.min_ms())
                .fold(f64::INFINITY, f64::min);
            let rlx_best = results
                .iter()
                .filter(|r| r.engine == "RLX" && r.n_channels == ch)
                .map(|r| r.min_ms())
                .fold(f64::INFINITY, f64::min);
            if burn_best.is_finite() && rlx_best.is_finite() {
                println!(
                    "  {ch:>3} ch: {:.2}x  (Burn {:.1} ms vs RLX {:.1} ms)",
                    burn_best / rlx_best,
                    burn_best,
                    rlx_best
                );
            }
        }
    }

    println!();
    println!("── CSV ──");
    if show_parity {
        println!("engine,backend,channels,time_samples,min_ms,mean_ms,parity_max_abs");
        for r in &results {
            let p = r
                .parity_max_abs
                .map(|v| format!("{v:.6}"))
                .unwrap_or_default();
            println!(
                "{},{},{},{},{:.1},{:.1},{}",
                r.engine,
                r.backend,
                r.n_channels,
                r.n_samples,
                r.min_ms(),
                r.mean_ms(),
                p,
            );
        }
    } else {
        println!("engine,backend,channels,time_samples,min_ms,mean_ms");
        for r in &results {
            println!(
                "{},{},{},{},{:.1},{:.1}",
                r.engine,
                r.backend,
                r.n_channels,
                r.n_samples,
                r.min_ms(),
                r.mean_ms(),
            );
        }
    }

    Ok(())
}
