/// Feed-Forward Network for LUNA transformer blocks.
///
/// Python: `FeedForwardBlock` in rope_transformer_encoder_block.py:
///   fc1(dim → hidden_dim) → GELU → Dropout → LayerNorm → fc2(hidden_dim → dim) → Dropout
///
/// At inference we skip dropout.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use crate::model::norm::LunaLayerNorm;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub fc1:  Linear<B>,
    pub fc2:  Linear<B>,
    pub norm: LunaLayerNorm<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize, norm_eps: f64, device: &B::Device) -> Self {
        Self {
            fc1:  LinearConfig::new(dim, hidden_dim).with_bias(true).init(device),
            fc2:  LinearConfig::new(hidden_dim, dim).with_bias(true).init(device),
            norm: LunaLayerNorm::new(hidden_dim, norm_eps, device),
        }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.fc1.forward(x));
        let h = self.norm.forward(h);
        self.fc2.forward(h)
    }
}
