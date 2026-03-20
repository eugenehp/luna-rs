//! Shared utilities for LUNA examples.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Instant;
use std::io::Write as _;

use anyhow::Context;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const DEFAULT_REPO: &str = "thorir/LUNA";

/// LUNA variant weight files on HuggingFace.
pub const VARIANT_FILES: &[(&str, &str)] = &[
    ("base",  "LUNA_base.safetensors"),
    ("large", "LUNA_large.safetensors"),
    ("huge",  "LUNA_huge.safetensors"),
];
pub const CONFIG_FILE: &str = "config.json";

/// Standard 22-channel TUEG bipolar montage.
pub const TUEG_CHANNELS: &[&str] = &[
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "A1-T3", "T4-A2",
];

// ── LUNA model configs (hardcoded to match Python YAML) ──────────────────────

use luna_rs::ModelConfig;

pub fn config_for_variant(variant: &str) -> ModelConfig {
    match variant {
        "base" => ModelConfig {
            patch_size: 40, num_queries: 4, embed_dim: 64,
            depth: 8, num_heads: 2, mlp_ratio: 4.0,
            num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
        },
        "large" => ModelConfig {
            patch_size: 40, num_queries: 6, embed_dim: 96,
            depth: 10, num_heads: 3, mlp_ratio: 4.0,
            num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
        },
        "huge" => ModelConfig {
            patch_size: 40, num_queries: 8, embed_dim: 128,
            depth: 24, num_heads: 4, mlp_ratio: 4.0,
            num_classes: 0, drop_path: 0.0, norm_eps: 1e-5,
        },
        _ => panic!("Unknown LUNA variant: {variant}. Use base, large, or huge."),
    }
}

pub fn weights_file_for_variant(variant: &str) -> &'static str {
    VARIANT_FILES.iter()
        .find(|(v, _)| *v == variant)
        .map(|(_, f)| *f)
        .unwrap_or_else(|| panic!("Unknown variant: {variant}"))
}

// ── HuggingFace weight resolution ─────────────────────────────────────────────

pub fn resolve_weights(
    repo_id: &str,
    variant: &str,
    weights_override: Option<&str>,
    cache_dir: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    if let Some(w) = weights_override {
        return Ok(w.into());
    }

    let filename = weights_file_for_variant(variant);

    #[cfg(feature = "hf-download")]
    {
        match hf_hub_resolve(repo_id, filename, cache_dir) {
            Ok(p) => return Ok(p),
            Err(e) => eprintln!("⚠  hf-hub: {e}  — falling back to cache scan"),
        }
    }

    match scan_hf_cache(repo_id, filename, cache_dir) {
        Ok(p) => return Ok(p),
        Err(_) => {}
    }

    println!("  Model not in local cache — downloading via Python huggingface_hub …");
    download_via_python(repo_id)?;
    scan_hf_cache(repo_id, filename, cache_dir)
}

#[cfg(feature = "hf-download")]
fn hf_hub_resolve(repo_id: &str, filename: &str, cache_dir: Option<&Path>) -> anyhow::Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let mut b = ApiBuilder::new().with_progress(true);
    if let Some(d) = cache_dir { b = b.with_cache_dir(d.to_path_buf()); }
    Ok(b.build()?.model(repo_id.to_string()).get(filename)?)
}

fn scan_hf_cache(repo_id: &str, filename: &str, cache_dir: Option<&Path>) -> anyhow::Result<PathBuf> {
    let base = cache_dir.map(PathBuf::from).unwrap_or_else(default_hf_cache);
    let snapshots = base
        .join(format!("models--{}", repo_id.replace('/', "--")))
        .join("snapshots");

    anyhow::ensure!(snapshots.exists(), "HF cache not found at {snapshots:?}");

    let mut dirs: Vec<_> = std::fs::read_dir(&snapshots)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    dirs.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    let snap = dirs.last().ok_or_else(|| anyhow::anyhow!("no snapshots"))?.path();

    let w = snap.join(filename);
    anyhow::ensure!(w.exists(), "weights not in snapshot: {w:?}");
    Ok(w)
}

fn download_via_python(repo_id: &str) -> anyhow::Result<()> {
    let python = find_python().ok_or_else(|| anyhow::anyhow!("no Python interpreter found"))?;
    let code = format!(
        "from huggingface_hub import snapshot_download; snapshot_download('{repo_id}')"
    );
    let out = std::process::Command::new(&python)
        .args(["-c", &code])
        .output()
        .with_context(|| format!("{python} failed"))?;
    std::io::stderr().write_all(&out.stderr).ok();
    if !out.status.success() {
        anyhow::bail!("snapshot_download failed");
    }
    Ok(())
}

fn find_python() -> Option<String> {
    for candidate in ["python3", "python"] {
        if std::process::Command::new(candidate).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(candidate.to_string());
        }
    }
    None
}

fn default_hf_cache() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HOME") { return PathBuf::from(v).join("hub"); }
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}

// ── EEG signal generation helpers ─────────────────────────────────────────────

/// Generate synthetic EEG: sum of sine waves at alpha/beta/theta bands + noise.
pub fn generate_synthetic_eeg(n_channels: usize, n_samples: usize, sfreq: f32) -> Vec<f32> {
    let mut signal = vec![0.0f32; n_channels * n_samples];
    for ch in 0..n_channels {
        let alpha_freq = 9.0 + (ch as f32 * 0.3);   // ~8-13 Hz
        let beta_freq = 18.0 + (ch as f32 * 0.5);    // ~13-30 Hz
        let theta_freq = 5.0 + (ch as f32 * 0.2);    // ~4-8 Hz
        // Simple LFSR-like deterministic "noise"
        let mut noise_state: u32 = (ch as u32 + 1) * 0xDEAD_BEEF;
        for t in 0..n_samples {
            let time = t as f32 / sfreq;
            let alpha = (2.0 * std::f32::consts::PI * alpha_freq * time).sin() * 20e-6;
            let beta  = (2.0 * std::f32::consts::PI * beta_freq * time).sin() * 5e-6;
            let theta = (2.0 * std::f32::consts::PI * theta_freq * time).sin() * 15e-6;
            noise_state ^= noise_state << 13;
            noise_state ^= noise_state >> 17;
            noise_state ^= noise_state << 5;
            let noise = (noise_state as f32 / u32::MAX as f32 - 0.5) * 2e-6;
            signal[ch * n_samples + t] = alpha + beta + theta + noise;
        }
    }
    signal
}

/// Get channel positions for the standard TUEG bipolar montage.
pub fn tueg_positions() -> Vec<f32> {
    TUEG_CHANNELS.iter()
        .flat_map(|name| {
            luna_rs::bipolar_channel_xyz(name)
                .unwrap_or([0.0, 0.0, 0.0])
                .to_vec()
        })
        .collect()
}

// ── Backend helpers for examples ─────────────────────────────────────────────

pub fn cpu_backend_name() -> &'static str {
    if cfg!(feature = "blas-accelerate") { "CPU (NdArray + Apple Accelerate)" }
    else if cfg!(feature = "openblas-system") { "CPU (NdArray + OpenBLAS)" }
    else { "CPU (NdArray + Rayon)" }
}

pub fn gpu_backend_name() -> &'static str {
    if cfg!(feature = "metal") { "GPU (wgpu — Metal / MSL)" }
    else if cfg!(feature = "vulkan") { "GPU (wgpu — Vulkan / SPIR-V)" }
    else { "GPU (wgpu — WGSL)" }
}

/// Run a closure with the appropriate backend based on compiled features.
///
/// Uses wgpu (GPU) when the `wgpu` feature is enabled, otherwise NdArray (CPU).
pub fn with_backend<F>(f: F) -> anyhow::Result<()>
where
    F: FnOnce(&str, Box<dyn std::any::Any>) -> anyhow::Result<()>,
{
    #[cfg(feature = "wgpu")]
    {
        let device = burn::backend::wgpu::WgpuDevice::default();
        f(gpu_backend_name(), Box::new(device))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        f(cpu_backend_name(), Box::new(device))
    }
}

/// Helper macro to dispatch `run::<B>(device, ...)` based on compiled backend features.
#[macro_export]
macro_rules! dispatch_backend {
    ($run_fn:ident, $($extra_args:expr),* $(,)?) => {{
        #[cfg(feature = "wgpu")]
        {
            type B = burn::backend::Wgpu;
            let device = burn::backend::wgpu::WgpuDevice::default();
            $run_fn::<B>(device, $($extra_args),*)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            type B = burn::backend::NdArray;
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            $run_fn::<B>(device, $($extra_args),*)
        }
    }};
}

// ── Step timer ────────────────────────────────────────────────────────────────

pub struct StepTimer {
    run_start:  Instant,
    step_start: Instant,
    step:       usize,
    total:      usize,
    pub verbose: bool,
}

impl StepTimer {
    pub fn new(total: usize, verbose: bool) -> Self {
        let now = Instant::now();
        Self { run_start: now, step_start: now, step: 0, total, verbose }
    }

    pub fn begin(&mut self, desc: &str) {
        self.step += 1;
        self.step_start = Instant::now();
        if self.verbose {
            println!("\n[{:>6.2}s] ▶ [{}/{}] {desc}", self.run_start.elapsed().as_secs_f64(), self.step, self.total);
        } else {
            print!("[{}/{}] {desc} … ", self.step, self.total);
            let _ = std::io::stdout().flush();
        }
    }

    pub fn done(&self, detail: &str) -> f64 {
        let ms = self.step_start.elapsed().as_secs_f64() * 1000.0;
        if self.verbose {
            println!("[{:>6.2}s] ✓  {ms:.0} ms  {detail}", self.run_start.elapsed().as_secs_f64());
        } else {
            println!("{ms:.0} ms  {detail}");
        }
        ms
    }

    pub fn sub(&self, msg: &str) {
        if self.verbose { println!("[{:>6.2}s]   {msg}", self.run_start.elapsed().as_secs_f64()); }
    }
}
