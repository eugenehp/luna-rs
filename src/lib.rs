//! # luna-rs — LUNA EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the LUNA (Latent Unified Network Architecture)
//! EEG foundation model. Two inference engines are available behind Cargo
//! features:
//!
//! | feature | module | runtime |
//! |---------|--------|---------|
//! | `burn`  | crate root (`LunaEncoder<B>`, …) | [Burn](https://burn.dev) 0.20 |
//! | `rlx`   | [`rlx`] | [RLX](https://docs.rs/rlx) compiler/runtime |
//!
//! LUNA uses cross-attention with learned queries to compress variable-channel
//! inputs into a fixed-size latent space, then processes them with a Rotary
//! Transformer encoder.
//!
//! ## Backends
//!
//! **Burn** (with `--features burn,ndarray`): `ndarray`, `blas-accelerate`,
//! `wgpu`, `metal`, `vulkan`.
//!
//! **RLX** (default): `cpu`, `metal`, `mlx`, `gpu`, `cuda`, `rocm`, `tpu`,
//! and BLAS variants.

#[cfg(not(any(feature = "burn", feature = "rlx")))]
compile_error!("enable at least one inference engine: `rlx` (default) and/or `burn`");

/// Configure the global Rayon thread pool (Burn NdArray + RLX CPU).
pub fn init_threads(n: Option<usize>) -> usize {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(count) = n {
        if count > 0 {
            builder = builder.num_threads(count);
        }
    }
    let _ = builder.build_global();
    rayon::current_num_threads()
}

pub mod channel_positions;
pub mod channel_vocab;
pub mod config;
#[cfg(feature = "burn")]
pub mod csv_loader;

#[cfg(feature = "burn")]
pub mod data;

#[cfg(feature = "burn")]
pub mod preprocessing;

#[cfg(feature = "burn")]
pub mod encoder;

#[cfg(feature = "burn")]
pub mod model;

#[cfg(feature = "burn")]
pub mod quantize;

#[cfg(feature = "burn")]
pub mod weights;

#[cfg(feature = "rlx")]
pub mod rlx;

// ── Burn re-exports (crate root) ─────────────────────────────────────────────

#[cfg(feature = "burn")]
pub use encoder::{EpochEmbedding, EncodingResult, LunaEncoder};

#[cfg(feature = "burn")]
pub use config::{DataConfig, ModelConfig};

#[cfg(feature = "burn")]
pub use data::{build_batch_named, FifInfo, InputBatch};

// When Burn is off, lift the RLX API to the crate root (default build).
#[cfg(all(feature = "rlx", not(feature = "burn")))]
pub use rlx::{
    load_edf as rlx_load_edf, load_fif as rlx_load_fif, save_epochs as rlx_save_epochs,
    EpochEmbedding, LunaEncoder, PreprocInfo as RlxPreprocInfo, RlxEpoch,
};

// ── Shared types ───────────────────────────────────────────────────────────

#[cfg(all(feature = "rlx", not(feature = "burn")))]
pub use config::ModelConfig;

pub use channel_positions::{
    bipolar_channel_xyz, channel_xyz, montage_channels, nearest_channel, normalise,
    MontageLayout,
};

pub use channel_vocab::{
    channel_index, channel_indices, channel_indices_unwrap, CHANNEL_VOCAB, SEED_CHANNELS,
    SIENA_CHANNELS, TUEG_CHANNELS, VOCAB_SIZE,
};

#[cfg(feature = "burn")]
pub use csv_loader::{load_from_csv, CsvInfo};

#[cfg(feature = "burn")]
pub use preprocessing::{
    load_csv_and_preprocess, load_edf, load_fif, load_luna_epochs, PreprocInfo,
};

#[cfg(test)]
#[cfg(feature = "burn")]
mod repeat_dim_test {
    use burn::backend::NdArray as B;
    use burn::prelude::*;

    #[test]
    fn test_repeat_dim_matches_pytorch() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let data = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
        let x = Tensor::<B, 3>::from_data(TensorData::new(data, vec![2, 2, 2]), &device);

        let r = x.repeat_dim(0, 3);
        assert_eq!(r.dims(), [6, 2, 2]);
        let vals = r.into_data().to_vec::<f32>().unwrap();
        let expected = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4., 5., 6., 7., 8., 1., 2., 3., 4.,
            5., 6., 7., 8.,
        ];
        assert_eq!(vals, expected, "repeat_dim should match PyTorch .repeat()");
    }
}

#[cfg(test)]
#[cfg(feature = "burn")]
mod trace_forward_test {
    use burn::backend::NdArray as B;
    use burn::nn::conv::Conv2dConfig;
    use burn::prelude::*;

    #[test]
    fn test_conv2d_basic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let conv = Conv2dConfig::new([1, 16], [1, 3])
            .with_stride([1, 1])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 1))
            .with_bias(true)
            .init::<B>(&device);
        let x = Tensor::<B, 4>::ones([1, 1, 8, 40], &device);
        let y = conv.forward(x);
        assert_eq!(y.dims()[3], 40);
    }
}
