/// Rotary Transformer Encoder Block for LUNA.
///
/// Python: `RotaryTransformerBlock` in rope_transformer_encoder_block.py:
///   x = x + drop_path(attn(norm1(x), freqs))
///   x = x + drop_path(ffn(norm2(x)))

use burn::prelude::*;
use crate::model::norm::LunaLayerNorm;
use crate::model::attention::RotarySelfAttention;
use crate::model::feedforward::FeedForward;

#[derive(Module, Debug)]
pub struct RotaryEncoderBlock<B: Backend> {
    pub norm1: LunaLayerNorm<B>,
    pub attn:  RotarySelfAttention<B>,
    pub norm2: LunaLayerNorm<B>,
    pub mlp:   FeedForward<B>,
}

impl<B: Backend> RotaryEncoderBlock<B> {
    pub fn new(
        dim:        usize,
        n_heads:    usize,
        mlp_ratio:  f64,
        qkv_bias:   bool,
        norm_eps:   f64,
        device:     &B::Device,
    ) -> Self {
        let hidden_dim = (dim as f64 * mlp_ratio) as usize;
        Self {
            norm1: LunaLayerNorm::new(dim, norm_eps, device),
            attn:  RotarySelfAttention::new(dim, n_heads, qkv_bias, device),
            norm2: LunaLayerNorm::new(dim, norm_eps, device),
            mlp:   FeedForward::new(dim, hidden_dim, norm_eps, device),
        }
    }

    /// x:     [B, S, dim]
    /// freqs: [S, head_dim/2, 2, 2]
    /// Returns: [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>, freqs: Tensor<B, 4>) -> Tensor<B, 3> {
        let h = x.clone() + self.attn.forward(self.norm1.forward(x.clone()), freqs);
        h.clone() + self.mlp.forward(self.norm2.forward(h))
    }
}
