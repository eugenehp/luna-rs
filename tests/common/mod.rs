//! Shared helpers for integration tests (import with `mod common;`).

#![allow(dead_code)]

use std::path::PathBuf;

/// Message printed when LUNA-Base weights are not available locally.
pub const SKIP_NO_WEIGHTS: &str =
    "SKIP: set LUNA_WEIGHTS or cache thorir/LUNA (LUNA_base.safetensors)";

fn hf_hub_root() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HOME") {
        return PathBuf::from(v).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Resolve `LUNA_base.safetensors` from `LUNA_WEIGHTS` or the HuggingFace cache.
pub fn find_weights() -> Option<PathBuf> {
    if let Ok(w) = std::env::var("LUNA_WEIGHTS") {
        let p = PathBuf::from(w);
        return p.exists().then_some(p);
    }
    let snaps = hf_hub_root().join("models--thorir--LUNA/snapshots");
    if !snaps.exists() {
        return None;
    }
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    dirs.sort_by_key(|e| {
        e.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    let w = dirs.last()?.path().join("LUNA_base.safetensors");
    w.exists().then_some(w)
}

/// `tests/vectors/config.json`, created with LUNA-Base defaults if missing.
pub fn test_config_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/config.json");
    if !p.exists() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let cfg = serde_json::json!({
            "model": {
                "patch_size": 40, "num_queries": 4, "embed_dim": 64,
                "depth": 8, "num_heads": 2, "mlp_ratio": 4.0,
                "num_classes": 0, "norm_eps": 1e-5
            }
        });
        std::fs::write(&p, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
    }
    p
}

/// Deterministic microvolt-scale EEG for parity tests.
pub fn synthetic_eeg(n_ch: usize, n_t: usize) -> Vec<f32> {
    (0..n_ch * n_t)
        .map(|i| (i as f32 * 0.01).sin() * 1e-5)
        .collect()
}

pub fn diff_max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn diff_rmse(a: &[f32], b: &[f32]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x - *y) as f64;
            d * d
        })
        .sum();
    (sum / a.len() as f64).sqrt()
}
