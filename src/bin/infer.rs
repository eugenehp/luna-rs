//! LUNA EEG inference — thin CLI over [`luna_rs::rlx::LunaEncoder`].

use std::path::Path;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use luna_rs::rlx::{load_edf, load_fif, save_epochs, LunaEncoder};
use luna_rs::init_threads;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
    Mlx,
    Gpu,
    Cuda,
    Rocm,
    Tpu,
}

impl DeviceArg {
    fn into_rlx(self) -> rlx::Device {
        match self {
            Self::Cpu => rlx::Device::Cpu,
            Self::Metal => rlx::Device::Metal,
            Self::Mlx => rlx::Device::Mlx,
            Self::Gpu => rlx::Device::Gpu,
            Self::Cuda => rlx::Device::Cuda,
            Self::Rocm => rlx::Device::Rocm,
            Self::Tpu => rlx::Device::Tpu,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "LUNA EEG model inference (RLX runtime)")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: DeviceArg,

    #[arg(long, env = "LUNA_WEIGHTS")]
    weights: String,

    #[arg(long, env = "LUNA_CONFIG")]
    config: String,

    /// Raw EEG recording (.edf or .fif). Omit to run synthetic zeros.
    #[arg(long)]
    input: Option<String>,

    #[arg(long)]
    output: String,

    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n_threads = init_threads(args.threads);
    let device = args.device.into_rlx();
    let t0 = Instant::now();

    eprintln!("Device   : {:?}  ({n_threads} threads)", device);

    let (mut model, ms_load) = LunaEncoder::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        device,
    )?;
    eprintln!("Model    : {}  ({ms_load:.0} ms)", model.describe());

    let t_pp = Instant::now();
    let (epochs, info) = if let Some(ref path) = args.input {
        let p = Path::new(path);
        if p.extension().and_then(|s| s.to_str()) == Some("edf") {
            load_edf(p)?
        } else {
            load_fif(p)?
        }
    } else {
        synthetic_epoch()?
    };
    let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Input    : {} epochs × {} ch × {} samples",
        epochs.len(),
        info.n_channels,
        epochs.first().map(|e| e.n_samples).unwrap_or(0),
    );
    eprintln!("Preproc  : {ms_preproc:.1} ms");

    let t_inf = Instant::now();
    let mut outputs = Vec::with_capacity(epochs.len());
    for ep in &epochs {
        outputs.push(model.run_rlx_epoch(ep)?);
    }
    let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
    let n = outputs.len();
    eprintln!(
        "Infer    : {ms_infer:.0} ms  ({n} epochs; {:.0} ms/epoch)",
        if n > 0 { ms_infer / n as f64 } else { 0.0 },
    );

    if args.verbose {
        for (i, ep) in outputs.iter().enumerate() {
            let mean: f64 =
                ep.output.iter().map(|&v| v as f64).sum::<f64>() / ep.output.len() as f64;
            eprintln!("  [ep {}] shape={:?}  mean={mean:+.4}", i + 1, ep.shape);
        }
    }

    save_epochs(&outputs, &args.output)?;
    eprintln!(
        "Output   → {}  ({:.0} ms total)",
        args.output,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn synthetic_epoch() -> anyhow::Result<(Vec<luna_rs::rlx::RlxEpoch>, luna_rs::rlx::PreprocInfo)> {
    use luna_rs::{channel_positions, rlx::RlxEpoch};

    let n_channels = 22usize;
    let n_samples = 1280usize;
    let names = [
        "FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2", "T3-C3",
        "C3-CZ", "CZ-C4", "C4-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
        "C4-P4", "P4-O2", "A1-T3", "T4-A2",
    ];
    let positions: Vec<f32> = names
        .iter()
        .flat_map(|n| channel_positions::bipolar_channel_xyz(n).unwrap_or([0.0, 0.0, 0.0]).to_vec())
        .collect();
    let channel_indices: Vec<i32> = names
        .iter()
        .map(|n| luna_rs::channel_index(n).unwrap_or(0) as i32)
        .collect();

    Ok((
        vec![RlxEpoch {
            signal: vec![0.0; n_channels * n_samples],
            chan_pos: positions,
            channel_indices: Some(channel_indices),
            n_channels,
            n_samples,
        }],
        luna_rs::rlx::PreprocInfo {
            ch_names: names.iter().map(|s| s.to_string()).collect(),
            n_channels,
            n_epochs: 1,
            target_sfreq: 256.0,
            epoch_dur: 5.0,
        },
    ))
}
