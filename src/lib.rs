//! # luna-rs — LUNA EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the LUNA (Latent Unified Network Architecture)
//! EEG foundation model, built on [Burn 0.20](https://burn.dev).
//!
//! LUNA is a topology-agnostic EEG model that uses cross-attention with
//! learned queries to compress variable-channel inputs into a fixed-size
//! latent space, then processes them with a Rotary Transformer encoder.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use luna_rs::LunaEncoder;
//!
//! let (model, _ms) = LunaEncoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! ```

pub mod channel_positions;
pub mod channel_vocab;
pub mod config;
pub mod csv_loader;
pub mod data;
pub mod encoder;
pub mod model;
pub mod preprocessing;
pub mod quantize;
pub mod weights;

// Flat re-exports
pub use encoder::{LunaEncoder, EpochEmbedding, EncodingResult};
pub use config::{ModelConfig, DataConfig};
pub use data::{InputBatch, FifInfo, build_batch_named};
pub use channel_positions::{channel_xyz, bipolar_channel_xyz, MontageLayout, montage_channels, nearest_channel, normalise};
pub use channel_vocab::{CHANNEL_VOCAB, VOCAB_SIZE, channel_index, channel_indices, channel_indices_unwrap, TUEG_CHANNELS, SIENA_CHANNELS, SEED_CHANNELS};
pub use csv_loader::{load_from_csv, CsvInfo};
pub use preprocessing::{load_edf, load_fif, load_luna_epochs, load_csv_and_preprocess, PreprocInfo};

#[cfg(test)]
mod repeat_dim_test {
    use burn::backend::NdArray as B;
    use burn::prelude::*;
    
    #[test]
    fn test_repeat_dim_matches_pytorch() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        // [2, 2, 2] tensor
        let data = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
        let x = Tensor::<B, 3>::from_data(TensorData::new(data, vec![2, 2, 2]), &device);
        
        let r = x.repeat_dim(0, 3); // repeat dim 0 by 3
        assert_eq!(r.dims(), [6, 2, 2]);
        let vals = r.into_data().to_vec::<f32>().unwrap();
        // PyTorch .repeat(3,1,1): [b0, b1, b0, b1, b0, b1]
        // = [1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8, 1,2,3,4, 5,6,7,8]
        let expected = vec![1.,2.,3.,4., 5.,6.,7.,8., 1.,2.,3.,4., 5.,6.,7.,8., 1.,2.,3.,4., 5.,6.,7.,8.];
        assert_eq!(vals, expected, "repeat_dim should match PyTorch .repeat()");
    }
}

#[cfg(test)]
mod trace_forward_test {
    use burn::backend::NdArray as B;
    use burn::prelude::*;

    #[test]
    fn test_conv2d_basic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        use burn::nn::conv::{Conv2dConfig};
        
        // Test simpler config first
        let conv = Conv2dConfig::new([1, 16], [1, 3])
            .with_stride([1, 1])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 1))
            .with_bias(true)
            .init::<B>(&device);
        let x = Tensor::<B, 4>::ones([1, 1, 8, 40], &device);
        eprintln!("Simple Conv2d input: {:?}", x.dims());
        let y = conv.forward(x);
        eprintln!("Simple Conv2d output: {:?}", y.dims());
        
        // Now test the problematic config — use manual padding instead
        // Conv2d(1, 16, (1, 19), stride=(1, 10), padding=(0, 9))
        // Pad the input manually, then use no-padding conv
        let conv2 = Conv2dConfig::new([1, 16], [1, 19])
            .with_stride([1, 10])
            .with_padding(burn::nn::PaddingConfig2d::Valid)
            .with_bias(true)
            .init::<B>(&device);
        let x2 = Tensor::<B, 4>::ones([1, 1, 8, 40], &device);
        // Manual pad: pad 9 on each side of W dim
        let pad_left = Tensor::<B, 4>::zeros([1, 1, 8, 9], &device);
        let pad_right = Tensor::<B, 4>::zeros([1, 1, 8, 9], &device);
        let x2_padded = Tensor::cat(vec![pad_left, x2, pad_right], 3); // [1,1,8,58]
        eprintln!("Manual-padded Conv2d input: {:?}", x2_padded.dims());
        let y2 = conv2.forward(x2_padded);
        eprintln!("Manual-padded Conv2d output: {:?}", y2.dims());
    }
    
    #[test]
    fn test_patch_embed_only() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let pe = crate::model::patch_embed::PatchEmbedNetwork::<B>::new(64, 40, &device);
        
        let x = Tensor::<B, 3>::ones([1, 4, 80], &device).mul_scalar(0.1f32);
        eprintln!("PatchEmbed input: {:?}", x.dims());
        let y = pe.forward(x);
        eprintln!("PatchEmbed output: {:?}", y.dims());
    }
}
