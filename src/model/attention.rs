/// Self-Attention with 1-D RoPE for LUNA's temporal encoder.
///
/// Python: `RotarySelfAttentionBlock` in rope_transformer_encoder_block.py.
///   qkv = Linear(dim, 3*dim) → reshape → [3, B, H, S, D]
///   q, k rotated via RotaryEmbedding
///   attn = softmax(q @ k^T / sqrt(d)) @ v
///   output = wo(attn)

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use crate::model::rope::apply_rope;

#[derive(Module, Debug)]
pub struct RotarySelfAttention<B: Backend> {
    pub qkv:     Linear<B>,
    pub proj:    Linear<B>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> RotarySelfAttention<B> {
    pub fn new(
        dim: usize,
        n_heads: usize,
        qkv_bias: bool,
        device: &B::Device,
    ) -> Self {
        let head_dim = dim / n_heads;
        Self {
            qkv:  LinearConfig::new(dim, dim * 3).with_bias(qkv_bias).init(device),
            proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            n_heads,
            head_dim,
        }
    }

    /// x:     [B, S, dim]
    /// freqs: [S, head_dim/2, 2, 2]
    /// Returns: [B, S, dim]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        freqs: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);

        // QKV projection: [B, S, 3*dim] → [B, S, 3, H, D] → [3, B, H, S, D]
        let qkv = self.qkv.forward(x)
            .reshape([b, s, 3, h, dh])
            .swap_dims(0, 2);  // [3, B, S, H, D]

        let q = qkv.clone().narrow(0, 0, 1).reshape([b, s, h, dh]);
        let k = qkv.clone().narrow(0, 1, 1).reshape([b, s, h, dh]);
        let v = qkv.narrow(0, 2, 1).reshape([b, s, h, dh]);

        // Apply RoPE to Q and K
        let (q, k) = apply_rope(q, k, freqs);

        // Transpose for attention: [B, H, S, D]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Scaled dot-product attention
        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v);  // [B, H, S, D]

        // Reshape back: [B, S, dim]
        let out = out.swap_dims(1, 2).reshape([b, s, h * dh]);
        self.proj.forward(out)
    }
}
