//! RLX forward graph vs Burn when both use identical prepared tokens.
//!
//! ```text
//! cargo test --release --no-default-features \
//!     --features burn,rlx,ndarray,rlx-cpu \
//!     --test rlx_graph_parity -- --nocapture
//! ```

#![cfg(all(feature = "burn", feature = "rlx"))]

mod common;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::*;
use common::{diff_max_abs, diff_rmse, find_weights, synthetic_eeg, test_config_path, SKIP_NO_WEIGHTS};
use luna_rs::model::luna::LunaOutput;
use luna_rs::rlx::LunaEncoder as RlxEncoder;

type B = NdArray<f32>;

#[test]
fn rlx_graph_matches_burn_given_same_tokens() {
    let Some(weights) = find_weights() else {
        eprintln!("{SKIP_NO_WEIGHTS}");
        return;
    };

    let n_ch = luna_rs::TUEG_CHANNELS.len();
    let n_t = 1280usize;
    let dev = NdArrayDevice::Cpu;
    let config_path = test_config_path();

    let signal = synthetic_eeg(n_ch, n_t);
    let mut signal_norm = signal.clone();
    luna_rs::rlx::prepare::channel_wise_normalize(&mut signal_norm, n_ch, n_t);

    let batch = luna_rs::build_batch_named::<B>(signal, luna_rs::TUEG_CHANNELS, n_t, &dev);
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
    let names = batch.channel_names.clone().unwrap();
    let (x_tok_t, chan_loc_emb) = model.prepare_tokens(
        signal_t.clone(),
        batch.channel_locations.clone(),
        None,
    );
    let n_patches = n_t / cfg.patch_size;
    let mut dec_q_t = chan_loc_emb;
    if let (Some(emb), Some(names)) = (&model.channel_emb, batch.channel_names.clone()) {
        let ch_emb = emb.forward(names);
        dec_q_t = dec_q_t + ch_emb.repeat_dim(0, n_patches);
    }
    let x_tok: Vec<f32> = x_tok_t.into_data().to_vec::<f32>().unwrap();
    let dec_q: Vec<f32> = dec_q_t.into_data().to_vec::<f32>().unwrap();

    let burn_out = match model.forward(
        Tensor::<B, 3>::from_data(TensorData::new(signal_norm.clone(), vec![1, n_ch, n_t]), &dev),
        batch.channel_locations.clone(),
        None,
        Some(names),
        &rope,
    ) {
        LunaOutput::Reconstruction { x_reconstructed, .. } => x_reconstructed
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        _ => panic!("expected reconstruction"),
    };

    let (mut rlx_enc, _) = RlxEncoder::load(&config_path, &weights, rlx::Device::Cpu).unwrap();
    let rlx_out = rlx_enc
        .run_forward_prepared(&x_tok, &dec_q, 1, n_ch, n_t, &[])
        .unwrap()
        .output;

    let max_abs = diff_max_abs(&burn_out, &rlx_out);
    let rmse = diff_rmse(&burn_out, &rlx_out);
    eprintln!("→ RLX graph vs Burn (shared prepare): max_abs={max_abs:.8}  rmse={rmse:.8}");
    assert!(max_abs < 3e-6, "graph max_abs={max_abs:.8}");
    assert!(rmse < 5e-7, "graph rmse={rmse:.8}");
}
