//! Compare RLX CPU `prepare_tokens` against Burn `prepare_tokens` intermediates.
//!
//! ```text
//! cargo test --release --no-default-features \
//!     --features burn,rlx,ndarray,rlx-cpu \
//!     --test rlx_prepare_burn -- --nocapture
//! ```

#![cfg(all(feature = "burn", feature = "rlx"))]

mod common;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::*;
use common::{diff_max_abs, find_weights, synthetic_eeg, SKIP_NO_WEIGHTS};
use luna_rs::rlx::prepare::{patch_embed_cpu, prepare_tokens};
use luna_rs::rlx::weights::{build_prepare_params, load_safetensors};

type B = NdArray<f32>;

#[test]
fn rlx_prepare_matches_burn() {
    let Some(weights) = find_weights() else {
        eprintln!("{SKIP_NO_WEIGHTS}");
        return;
    };

    let dev = NdArrayDevice::Cpu;
    let n_ch = luna_rs::TUEG_CHANNELS.len();
    let n_t = 1280usize;
    let p = 40usize;
    let d = 64usize;

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

    let signal_t = Tensor::<B, 3>::from_data(
        TensorData::new(signal_norm.clone(), vec![1, n_ch, n_t]),
        &dev,
    );

    let mut raw = load_safetensors(weights.to_str().unwrap()).unwrap();
    let prep_params = build_prepare_params(&mut raw).unwrap();

    let burn_patch: Vec<f32> = model
        .patch_embed
        .forward(signal_t.clone())
        .squeeze::<2>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let rlx_patch = patch_embed_cpu(&signal_norm, 1, n_ch, n_t, p, d, &prep_params);
    assert_eq!(rlx_patch.len(), burn_patch.len(), "patch output length");
    let pe_err = diff_max_abs(&rlx_patch, &burn_patch);
    eprintln!(
        "patch_embed diff_max_abs={pe_err:.6}  len={}",
        burn_patch.len()
    );
    if pe_err > 1.0 {
        let mut worst = (0usize, 0.0f32);
        for (i, (a, b)) in rlx_patch.iter().zip(burn_patch.iter()).enumerate() {
            let d = (a - b).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        eprintln!("worst idx={} rlx={} burn={}", worst.0, rlx_patch[worst.0], burn_patch[worst.0]);
    }

    let burn_freq: Vec<f32> = model
        .freq_embed
        .forward(signal_t.clone())
        .squeeze::<2>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let burn_combined: Vec<f32> = burn_patch
        .iter()
        .zip(burn_freq.iter())
        .map(|(a, b)| a + b)
        .collect();
    let rlx_combined: Vec<f32> = rlx_patch
        .iter()
        .zip(
            luna_rs::rlx::prepare::freq_embed_cpu(&signal_norm, 1, n_ch, n_t, p, &prep_params)
                .iter(),
        )
        .map(|(a, b)| a + b)
        .collect();
    let comb_err = diff_max_abs(&rlx_combined, &burn_combined);
    eprintln!("patch+freq combined diff_max_abs={comb_err:.6}");

    let (burn_xtok, burn_dec) = model.prepare_tokens(
        signal_t.clone(),
        batch.channel_locations.clone(),
        None,
    );
    let burn_xtok: Vec<f32> = burn_xtok.into_data().to_vec::<f32>().unwrap();
    let burn_dec: Vec<f32> = burn_dec.into_data().to_vec::<f32>().unwrap();

    let chan_locs: Vec<f32> = batch
        .channel_locations
        .squeeze::<2>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let (rlx_xtok, rlx_dec) = prepare_tokens(
        &signal_norm,
        &chan_locs,
        None,
        1,
        n_ch,
        n_t,
        p,
        d,
        &prep_params,
    );

    let xtok_err = diff_max_abs(&rlx_xtok, &burn_xtok);
    let dec_err = diff_max_abs(&rlx_dec, &burn_dec);
    eprintln!("x_tokenized diff_max_abs={xtok_err:.6}");
    eprintln!("decoder_queries diff_max_abs={dec_err:.6}");
    let burn_freq_only: Vec<f32> = model
        .freq_embed
        .forward(signal_t)
        .squeeze::<2>()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let rlx_freq_only =
        luna_rs::rlx::prepare::freq_embed_cpu(&signal_norm, 1, n_ch, n_t, p, &prep_params);
    let fe_err = diff_max_abs(&rlx_freq_only, &burn_freq_only);
    eprintln!("freq_embed diff_max_abs={fe_err:.8}");

    assert!(pe_err < 1e-5, "patch_embed diff_max_abs={pe_err:.8}");
    assert!(fe_err < 1e-6, "freq_embed diff_max_abs={fe_err:.8}");
    assert!(comb_err < 1e-5, "patch+freq diff_max_abs={comb_err:.8}");
    assert!(xtok_err < 1e-5, "x_tokenized diff_max_abs={xtok_err:.8}");
    assert!(dec_err < 1e-6, "decoder_queries diff_max_abs={dec_err:.8}");
}
