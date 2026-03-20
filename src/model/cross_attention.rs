/// Cross-Attention Block for LUNA's channel unification.
///
/// Python: `CrossAttentionBlock` in LUNA.py.
///
/// Python uses:
///   self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
///   self.ffn = Mlp(input_embed_dim, ff_dim, output_embed_dim, GELU, drop, norm_layer=LayerNorm)
///   self.keys_norm = LayerNorm(D)
///   self.values_norm = LayerNorm(D)
///   self.queries_norm = LayerNorm(D)
///   self.query_self_attn = nn.TransformerEncoder(
///       nn.TransformerEncoderLayer(D, nhead, activation='gelu', dim_feedforward=ff_dim,
///                                  batch_first=True, norm_first=True),
///       num_layers=3)
///
/// nn.MultiheadAttention stores a fused in_proj_weight \[3*D, D\] and in_proj_bias \[3*D\],
/// plus out_proj.weight \[D, D\] and out_proj.bias \[D\].
///
/// nn.TransformerEncoderLayer (norm_first=True) structure:
///   norm1 → self_attn (nn.MultiheadAttention) → dropout → residual
///   norm2 → linear1 → activation → dropout → linear2 → dropout → residual

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use crate::model::norm::LunaLayerNorm;

// ── MultiheadAttention (fused in_proj) ──────────────────────────────────────

/// Matches nn.MultiheadAttention with fused in_proj_weight/bias.
/// Weight key: in_proj_weight \[3*D, D\], in_proj_bias \[3*D\], out_proj.weight \[D,D\], out_proj.bias \[D\]
#[derive(Module, Debug)]
pub struct FusedMultiheadAttention<B: Backend> {
    pub in_proj:  Linear<B>,   // [D, 3*D] (burn stores transposed)
    pub out_proj: Linear<B>,   // [D, D]
    pub n_heads:  usize,
    pub head_dim: usize,
}

impl<B: Backend> FusedMultiheadAttention<B> {
    pub fn new(dim: usize, n_heads: usize, device: &B::Device) -> Self {
        let head_dim = dim / n_heads;
        Self {
            in_proj:  LinearConfig::new(dim, dim * 3).with_bias(true).init(device),
            out_proj: LinearConfig::new(dim, dim).with_bias(true).init(device),
            n_heads,
            head_dim,
        }
    }

    /// q_input, k_input, v_input: [B, S_q, D] / [B, S_kv, D]
    /// Returns: (output [B, S_q, D], attn_weights [B, S_q, S_kv])
    pub fn forward(
        &self,
        q_input: Tensor<B, 3>,
        k_input: Tensor<B, 3>,
        v_input: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s_q, _] = q_input.dims();
        let s_kv = k_input.dims()[1];
        let (h, dh) = (self.n_heads, self.head_dim);
        let dim = h * dh;

        // Fused projection for Q
        let qkv_q = self.in_proj.forward(q_input);  // [B, S_q, 3*D]
        let q = qkv_q.narrow(2, 0, dim).reshape([b, s_q, h, dh]).swap_dims(1, 2);

        // Fused projection for K
        let qkv_k = self.in_proj.forward(k_input);
        let k = qkv_k.narrow(2, dim, dim).reshape([b, s_kv, h, dh]).swap_dims(1, 2);

        // Fused projection for V
        let qkv_v = self.in_proj.forward(v_input);
        let v = qkv_v.narrow(2, dim * 2, dim).reshape([b, s_kv, h, dh]).swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn_weights = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);  // [B, H, S_q, S_kv]

        let out = attn_weights.clone().matmul(v);  // [B, H, S_q, D]
        let out = self.out_proj.forward(
            out.swap_dims(1, 2).reshape([b, s_q, dim])
        );

        // Average attention weights across heads: [B, H, S_q, S_kv] → [B, S_q, S_kv]
        let avg_attn = attn_weights.mean_dim(1).squeeze::<3>();

        (out, avg_attn)
    }
}

// ── TransformerEncoderLayer (norm_first=True) ───────────────────────────────

/// Matches nn.TransformerEncoderLayer(norm_first=True, activation='gelu').
/// Structure:
///   x = x + self_attn(norm1(x))
///   x = x + ffn(norm2(x))
/// where ffn = linear1 → gelu → linear2
#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    pub norm1:     LunaLayerNorm<B>,
    pub self_attn: FusedMultiheadAttention<B>,
    pub norm2:     LunaLayerNorm<B>,
    pub linear1:   Linear<B>,
    pub linear2:   Linear<B>,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    pub fn new(dim: usize, n_heads: usize, ff_dim: usize, norm_eps: f64, device: &B::Device) -> Self {
        Self {
            norm1:     LunaLayerNorm::new(dim, norm_eps, device),
            self_attn: FusedMultiheadAttention::new(dim, n_heads, device),
            norm2:     LunaLayerNorm::new(dim, norm_eps, device),
            linear1:   LinearConfig::new(dim, ff_dim).with_bias(true).init(device),
            linear2:   LinearConfig::new(ff_dim, dim).with_bias(true).init(device),
        }
    }

    /// x: [B, S, D] → [B, S, D]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // norm_first: norm before attention
        let normed = self.norm1.forward(x.clone());
        let (attn_out, _) = self.self_attn.forward(normed.clone(), normed.clone(), normed);
        let x = x + attn_out;

        // FFN with norm_first
        let normed = self.norm2.forward(x.clone());
        let ff_out = self.linear2.forward(gelu(self.linear1.forward(normed)));
        x + ff_out
    }
}

// ── CrossAttentionBlock ─────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    /// Learnable query prototypes: [1, Q, D].
    pub query_embed: Param<Tensor<B, 3>>,
    /// Temperature parameter (stored but unused in forward).
    pub temperature: Param<Tensor<B, 1>>,
    /// Cross-attention: nn.MultiheadAttention (fused in_proj)
    pub cross_attention: FusedMultiheadAttention<B>,
    /// FFN: timm.Mlp(D, ff_dim, D, GELU, drop, norm_layer=LayerNorm)
    /// At inference: fc1 → GELU → LayerNorm → fc2
    pub ffn_fc1:  Linear<B>,
    pub ffn_norm: LunaLayerNorm<B>,
    pub ffn_fc2:  Linear<B>,
    /// Pre-attention norms
    pub queries_norm: LunaLayerNorm<B>,
    pub keys_norm:    LunaLayerNorm<B>,
    pub values_norm:  LunaLayerNorm<B>,
    /// 3-layer TransformerEncoder for query self-attention
    pub self_attn_layers: Vec<TransformerEncoderLayer<B>>,
    pub num_queries: usize,
}

impl<B: Backend> CrossAttentionBlock<B> {
    pub fn new(
        num_queries: usize,
        input_embed_dim: usize,
        output_embed_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let self_attn_layers = (0..3)
            .map(|_| TransformerEncoderLayer::new(
                input_embed_dim, num_heads, ff_dim, norm_eps, device,
            ))
            .collect();

        Self {
            query_embed: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, num_queries, input_embed_dim], device),
            ),
            temperature: Param::initialized(
                ParamId::new(),
                Tensor::ones([1], device),
            ),
            cross_attention: FusedMultiheadAttention::new(input_embed_dim, num_heads, device),
            ffn_fc1:  LinearConfig::new(input_embed_dim, ff_dim).with_bias(true).init(device),
            ffn_norm: LunaLayerNorm::new(ff_dim, norm_eps, device),
            ffn_fc2:  LinearConfig::new(ff_dim, output_embed_dim).with_bias(true).init(device),
            queries_norm: LunaLayerNorm::new(input_embed_dim, norm_eps, device),
            keys_norm:    LunaLayerNorm::new(input_embed_dim, norm_eps, device),
            values_norm:  LunaLayerNorm::new(input_embed_dim, norm_eps, device),
            self_attn_layers,
            num_queries,
        }
    }

    /// x: [B_eff, C, D] — channel tokens for one time patch
    /// Returns: ([B_eff, Q, D], [B_eff, Q, C]) — unified queries + attention scores
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch_size, _num_channels, dim] = x.dims();

        // Expand learnable queries: [1, Q, D] → [B_eff, Q, D]
        let queries = self.query_embed.val()
            .expand([batch_size, self.num_queries, dim]);

        let queries = self.queries_norm.forward(queries);
        let keys = self.keys_norm.forward(x.clone());
        let values = self.values_norm.forward(x);

        // Cross-attention via nn.MultiheadAttention
        let (attention_out, attention_scores) =
            self.cross_attention.forward(queries, keys, values);
        // attention_out: [B_eff, Q, D], attention_scores: [B_eff, Q, C]

        // FFN with residual: timm.Mlp(D, ff_dim, D, GELU, norm_layer=LayerNorm)
        let ffn_out = self.ffn_fc2.forward(
            self.ffn_norm.forward(gelu(self.ffn_fc1.forward(attention_out.clone())))
        );
        let attention_out = ffn_out + attention_out;

        // 3-layer TransformerEncoder for query self-attention
        let mut out = attention_out;
        for layer in &self.self_attn_layers {
            out = layer.forward(out);
        }

        (out, attention_scores)
    }
}
