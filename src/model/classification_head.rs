/// Classification Head with Queries for LUNA fine-tuning.
///
/// Python: `ClassificationHeadWithQueries` in LUNA.py.
///
/// Python uses:
///   self.embed_dim = embed_dim * num_queries  (= Q*D)
///   self.decoder_attn = nn.MultiheadAttention(Q*D, num_heads, batch_first=True, dropout=0.15)
///   self.decoder_ffn = Mlp(Q*D, Q*D*4, num_classes, GELU, drop=0.15)
///   self.learned_agg = nn.Parameter(torch.randn(1, 1, Q*D))
///
/// Forward:
///   decoder_queries = learned_agg.repeat(B, 1, 1)
///   x = decoder_attn(query=decoder_queries, key=x, value=x)[0]
///   x = x[:, 0, :]  # take first token
///   x = decoder_ffn(x)
///   return x  # [B, num_classes]
///
/// At inference, dropout is disabled, so:
///   MultiheadAttention → regular attention (no dropout)
///   Mlp → fc1 → GELU → fc2  (no dropout, no norm since norm_layer=None by default)

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use crate::model::cross_attention::FusedMultiheadAttention;

#[derive(Module, Debug)]
pub struct ClassificationHead<B: Backend> {
    /// Learned aggregation query: [1, 1, Q*D].
    pub learned_agg: Param<Tensor<B, 3>>,
    /// nn.MultiheadAttention(Q*D, num_heads) — fused in_proj
    pub decoder_attn: FusedMultiheadAttention<B>,
    /// timm.Mlp(Q*D, Q*D*4, num_classes, GELU, drop=0.15)
    /// At inference: fc1(Q*D → 4*Q*D) → GELU → fc2(4*Q*D → num_classes)
    pub ffn_fc1: Linear<B>,
    pub ffn_fc2: Linear<B>,
    pub full_dim: usize,
}

impl<B: Backend> ClassificationHead<B> {
    pub fn new(
        embed_dim: usize,
        num_queries: usize,
        num_heads: usize,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        let full_dim = embed_dim * num_queries;

        Self {
            learned_agg: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, 1, full_dim], device),  // will be overwritten by weights
            ),
            decoder_attn: FusedMultiheadAttention::new(full_dim, num_heads, device),
            ffn_fc1: LinearConfig::new(full_dim, full_dim * 4).with_bias(true).init(device),
            ffn_fc2: LinearConfig::new(full_dim * 4, num_classes).with_bias(true).init(device),
            full_dim,
        }
    }

    /// x: [B, num_patches, Q*D] → [B, num_classes]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, _n, _d] = x.dims();

        // Expand aggregation query: [1, 1, Q*D] → [B, 1, Q*D]
        let queries = self.learned_agg.val()
            .expand([b, 1, self.full_dim]);

        // MultiheadAttention: query=learned_agg, key=x, value=x
        let (attn_out, _) = self.decoder_attn.forward(queries, x.clone(), x);
        // attn_out: [B, 1, Q*D]

        // Take first (only) token: x[:, 0, :]
        let out = attn_out.narrow(1, 0, 1).reshape([b, self.full_dim]);

        // Mlp: fc1 → GELU → fc2
        let out = gelu(self.ffn_fc1.forward(out));
        let out = self.ffn_fc2.forward(out);  // [B, num_classes]

        out.unsqueeze_dim::<3>(1)  // [B, 1, num_classes]
    }
}
