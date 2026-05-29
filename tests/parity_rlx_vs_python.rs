//! Parity: RLX `LunaEncoder` vs Python reference vectors.
//!
//! ```text
//! python scripts/export_parity_vectors.py
//! cargo test --release --test parity_rlx_vs_python -- --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{diff_max_abs, diff_rmse, find_weights, test_config_path, SKIP_NO_WEIGHTS};
use luna_rs::rlx::{LunaEncoder, RunEpochOpts};

fn vectors_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/parity.safetensors")
}

fn load_f32_2d(st: &safetensors::SafeTensors, key: &str) -> Vec<f32> {
    let view = st.tensor(key).unwrap_or_else(|_| panic!("missing key {key}"));
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_i32_1d(st: &safetensors::SafeTensors, key: &str) -> Vec<i32> {
    let view = st.tensor(key).unwrap_or_else(|_| panic!("missing key {key}"));
    view.data()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn pick_rlx_device() -> rlx::Device {
    match std::env::var("LUNA_RLX_DEVICE").as_deref().unwrap_or("cpu") {
        "cpu" => rlx::Device::Cpu,
        "metal" => rlx::Device::Metal,
        "mlx" => rlx::Device::Mlx,
        other => panic!("unknown LUNA_RLX_DEVICE: {other:?}"),
    }
}

#[test]
fn rlx_matches_python_reference() {
    let vpath = vectors_path();
    if !vpath.exists() {
        eprintln!("SKIP: parity vectors not found at {}", vpath.display());
        eprintln!("  Run: python scripts/export_parity_vectors.py");
        return;
    }
    let Some(weights) = find_weights() else {
        eprintln!("{SKIP_NO_WEIGHTS}");
        return;
    };

    let bytes = std::fs::read(&vpath).expect("read vectors");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse vectors");

    let input_norm = load_f32_2d(&st, "input_normalized");
    let chan_locs = load_f32_2d(&st, "channel_locations");
    let chan_names = load_i32_1d(&st, "channel_names");
    let py_output = load_f32_2d(&st, "output_reconstructed");

    let n_ch = 22usize;
    let n_t = 1280usize;
    assert_eq!(input_norm.len(), n_ch * n_t);
    assert_eq!(py_output.len(), n_ch * n_t);

    let config_path = test_config_path();
    let (mut enc, _) = LunaEncoder::load(&config_path, &weights, pick_rlx_device())
        .expect("rlx load");

    let rlx_out = enc
        .run_epoch_opts(
            &input_norm,
            &chan_locs,
            Some(&chan_names),
            n_ch,
            n_t,
            RunEpochOpts {
                normalize: false,
            },
        )
        .expect("rlx run")
        .output;

    assert_eq!(rlx_out.len(), py_output.len());
    let max_abs = diff_max_abs(&rlx_out, &py_output);
    let rmse = diff_rmse(&rlx_out, &py_output);
    eprintln!("→ RLX vs Python: max_abs={max_abs:.6}  rmse={rmse:.6}");
    assert!(max_abs < 1e-4, "max_abs {max_abs:.6} > 1e-4");
    assert!(rmse < 1e-5, "rmse {rmse:.6} > 1e-5");
}
