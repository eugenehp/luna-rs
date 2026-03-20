/// LayerNorm wrapper for LUNA.
///
/// LUNA uses standard LayerNorm (not RMSNorm like ZUNA).
/// burn 0.20.1 provides `burn::nn::LayerNorm` natively.

use burn::prelude::*;
use burn::nn::{LayerNorm, LayerNormConfig};

#[derive(Module, Debug)]
pub struct LunaLayerNorm<B: Backend> {
    pub inner: LayerNorm<B>,
}

impl<B: Backend> LunaLayerNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            inner: LayerNormConfig::new(dim).with_epsilon(eps).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x)
    }
}
