/// Patch Embedding Network for LUNA.
///
/// Python: `PatchEmbedNetwork` in LUNA.py.
///
/// Input:  (B, C, T)
/// Step 1: rearrange to (B, C*S, P) where S = T/patch_size, P = patch_size
/// Step 2: unsqueeze to (B, 1, C*S, P) — treat as 2D image
/// Step 3: 3-layer CNN:
///   Conv2d(1, out_ch, (1, kernel-1), stride=(1, kernel//2), pad=(0, kernel//2-1))
///   → GroupNorm → GELU
///   Conv2d(out_ch, out_ch, (1,3), stride=1, pad=(0,1)) → GroupNorm → GELU
///   Conv2d(out_ch, out_ch, (1,3), stride=1, pad=(0,1)) → GroupNorm → GELU
/// Step 4: rearrange to (B, C*S, D)
///
/// Where out_ch = embed_dim // 4, kernel = patch_size // 2

use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    GroupNorm, GroupNormConfig,
};
use burn::tensor::activation::gelu;

#[derive(Module, Debug)]
pub struct PatchEmbedNetwork<B: Backend> {
    pub conv1: Conv2d<B>,
    pub gn1:   GroupNorm<B>,
    pub conv2: Conv2d<B>,
    pub gn2:   GroupNorm<B>,
    pub conv3: Conv2d<B>,
    pub gn3:   GroupNorm<B>,
    pub patch_size: usize,
    pub embed_dim:  usize,
}

impl<B: Backend> PatchEmbedNetwork<B> {
    pub fn new(embed_dim: usize, patch_size: usize, device: &B::Device) -> Self {
        let out_channels = embed_dim / 4;
        let groups = 4;
        let kernel = patch_size / 2;

        // First conv: downsample along time
        // NOTE: We use Valid padding and manually pad the input, because
        // Burn NdArray's Conv2d has a bug with Explicit padding for large kernels.
        let conv1 = Conv2dConfig::new([1, out_channels], [1, kernel - 1])
            .with_stride([1, kernel / 2])
            .with_padding(burn::nn::PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);

        let gn1 = GroupNormConfig::new(groups, out_channels)
            .with_epsilon(1e-5)
            .init(device);

        // Second conv: preserve shape
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [1, 3])
            .with_stride([1, 1])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 1))
            .with_bias(true)
            .init(device);

        let gn2 = GroupNormConfig::new(groups, out_channels)
            .with_epsilon(1e-5)
            .init(device);

        // Third conv: preserve shape
        let conv3 = Conv2dConfig::new([out_channels, out_channels], [1, 3])
            .with_stride([1, 1])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(0, 1))
            .with_bias(true)
            .init(device);

        let gn3 = GroupNormConfig::new(groups, out_channels)
            .with_epsilon(1e-5)
            .init(device);

        Self {
            conv1, gn1, conv2, gn2, conv3, gn3,
            patch_size, embed_dim,
        }
    }

    /// x: [B, C, T] → [B, C*S, D] where S = T/patch_size, D = embed_dim
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, t] = x.dims();
        let s = t / self.patch_size;

        // Rearrange: [B, C, T] → [B, C*S, P] → [B, 1, C*S, P]
        let x = x.reshape([b, c * s, self.patch_size]).unsqueeze_dim::<4>(1);

        // 3-layer CNN
        // Conv1 uses manual W-padding to work around a Burn NdArray backend bug
        let kernel = self.patch_size / 2;
        let w_pad = kernel / 2 - 1;
        let cs = c * s;
        let pad_left  = Tensor::zeros([b, 1, cs, w_pad], &x.device());
        let pad_right = Tensor::zeros([b, 1, cs, w_pad], &x.device());
        let x_padded = Tensor::cat(vec![pad_left, x, pad_right], 3);
        let x = gelu(self.gn1.forward(self.conv1.forward(x_padded)));
        let x = gelu(self.gn2.forward(self.conv2.forward(x)));
        let x = gelu(self.gn3.forward(self.conv3.forward(x)));

        // Python: rearrange(x, 'B E CS D -> B CS (D E)')
        // Output shape: [B, E, C*S, D'] → permute to [B, C*S, D', E] → flatten to [B, C*S, D'*E]
        // Note: Python's (D E) means D varies fastest (inner), E varies slowest (outer)
        // So we need [B, CS, D', E] then reshape, NOT [B, CS, E, D']
        let [b2, e, cs, d_prime] = x.dims();
        x.swap_dims(1, 2)            // [B, C*S, E, D']
         .swap_dims(2, 3)            // [B, C*S, D', E]
         .reshape([b2, cs, d_prime * e])  // [B, C*S, D'*E = embed_dim]
    }
}
