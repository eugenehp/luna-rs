//! Parity: RLX `LunaEncoder` vs Burn `Luna` forward pass on the same inputs.
//!
//! ```text
//! cargo test --release --no-default-features \
//!     --features burn,rlx,ndarray,rlx-cpu \
//!     --test parity_rlx_vs_burn -- --nocapture
//! ```

#![cfg(all(feature = "burn", feature = "rlx"))]

mod common;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::*;
use common::{diff_max_abs, diff_rmse, find_weights, synthetic_eeg, test_config_path, SKIP_NO_WEIGHTS};
use luna_rs::rlx::{LunaEncoder as RlxEncoder, RunEpochOpts};
use luna_rs::model::luna::LunaOutput;

type B = NdArray<f32>;

fn pick_rlx_device() -> rlx::Device {
    match std::env::var("LUNA_RLX_DEVICE").as_deref().unwrap_or("cpu") {
        "cpu" => rlx::Device::Cpu,
        "metal" => rlx::Device::Metal,
        "mlx" => rlx::Device::Mlx,
        other => panic!("unknown LUNA_RLX_DEVICE: {other:?}"),
    }
}

#[test]
fn rlx_encoder_matches_burn_encoder() {
    let Some(weights) = find_weights() else {
        eprintln!("{SKIP_NO_WEIGHTS}");
        return;
    };

    let n_ch = luna_rs::TUEG_CHANNELS.len();
    let n_t = 1280usize;
    let config_path = test_config_path();
    let dev = NdArrayDevice::Cpu;

    let signal = synthetic_eeg(n_ch, n_t);
    let signal_norm = {
        let mut s = signal.clone();
        luna_rs::rlx::prepare::channel_wise_normalize(&mut s, n_ch, n_t);
        s
    };

    let batch = luna_rs::build_batch_named::<B>(signal, luna_rs::TUEG_CHANNELS, n_t, &dev);
    let chan_locs: Vec<f32> = batch
        .channel_locations
        .clone()
        .squeeze::<2>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let chan_names_i32: Vec<i32> = batch
        .channel_names
        .clone()
        .unwrap()
        .into_data()
        .to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect();

    let cfg = luna_rs::ModelConfig::default();
    let model = luna_rs::weights::load_model::<B>(
        &cfg,
        weights.to_str().unwrap(),
        luna_rs::VOCAB_SIZE,
        &dev,
    )
    .unwrap();
    let rope = luna_rs::model::rope::RotaryEmbedding::<B>::new(cfg.head_dim(), 1024, 10_000.0, &dev);

    let signal_t = Tensor::<B, 3>::from_data(
        TensorData::new(signal_norm.clone(), vec![1, n_ch, n_t]),
        &dev,
    );
    let locations = batch.channel_locations.clone();
    let names = batch.channel_names.clone().unwrap();

    let burn_out = match model.forward(signal_t, locations, None, Some(names), &rope) {
        LunaOutput::Reconstruction {
            x_reconstructed, ..
        } => x_reconstructed
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        _ => panic!("expected reconstruction"),
    };

    let (mut rlx_enc, _) = RlxEncoder::load(&config_path, &weights, pick_rlx_device()).unwrap();
    let rlx_out = rlx_enc
        .run_epoch_opts(
            &signal_norm,
            &chan_locs,
            Some(&chan_names_i32),
            n_ch,
            n_t,
            RunEpochOpts {
                normalize: false,
            },
        )
        .unwrap()
        .output;

    assert_eq!(burn_out.len(), rlx_out.len());
    let max_abs = diff_max_abs(&burn_out, &rlx_out);
    let rmse = diff_rmse(&burn_out, &rlx_out);
    eprintln!("→ RLX vs Burn: max_abs={max_abs:.6}  rmse={rmse:.6}");
    assert!(max_abs < 3e-6, "parity failed: max_abs={max_abs:.8}");
    assert!(rmse < 5e-7, "parity failed: rmse={rmse:.8}");
}
