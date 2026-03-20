/// Patch Reconstruction Head with Queries for LUNA pre-training.
///
/// Python: `PatchReconstructionHeadWithQueries` in LUNA.py.
///
/// Python uses:
///   self.decoder_pred = nn.TransformerDecoder(
///       nn.TransformerDecoderLayer(embed_dim, num_heads, dropout=0.0,
///           batch_first=True, activation='gelu', dim_feedforward=embed_dim*4, norm_first=True),
///       num_layers=1)
///   self.norm = nn.LayerNorm(embed_dim)
///   self.decoder_linear = Mlp(embed_dim, embed_dim*4, input_dim, GELU, drop=0.0)
///
/// nn.TransformerDecoderLayer (norm_first=True) has 3 sub-layers:
///   1. self_attn:  norm → multihead_self_attn(tgt, tgt, tgt) → residual
///   2. cross_attn: norm → multihead_cross_attn(tgt, memory, memory) → residual
///   3. ffn:        norm → linear1 → gelu → linear2 → residual
///
/// Forward:
///   enc: [B, num_patches, Q*D]  →  reshape to [B*t, Q, D]
///   out = decoder_pred(decoder_queries, enc)  # tgt=queries, memory=enc
///   out = norm(out)
///   out = decoder_linear(out)  # [B*t, C, patch_size]
///   out = rearrange '(B t) C P -> B C (t P)'

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use crate::model::norm::LunaLayerNorm;
use crate::model::cross_attention::FusedMultiheadAttention;

#[derive(Module, Debug)]
pub struct ReconstructionHead<B: Backend> {
    // TransformerDecoderLayer sub-layer 1: self-attention on target (decoder queries)
    pub self_attn_norm: LunaLayerNorm<B>,
    pub self_attn:      FusedMultiheadAttention<B>,
    // TransformerDecoderLayer sub-layer 2: cross-attention (tgt attends to memory)
    pub cross_attn_norm: LunaLayerNorm<B>,
    pub cross_attn:      FusedMultiheadAttention<B>,
    // TransformerDecoderLayer sub-layer 3: FFN
    pub ffn_norm:    LunaLayerNorm<B>,
    pub ffn_linear1: Linear<B>,
    pub ffn_linear2: Linear<B>,
    // Post-decoder norm
    pub output_norm: LunaLayerNorm<B>,
    // Output MLP: timm.Mlp(embed_dim, embed_dim*4, patch_size, GELU, drop=0.0)
    // No norm_layer → at inference: fc1 → GELU → fc2
    pub output_fc1: Linear<B>,
    pub output_fc2: Linear<B>,
    pub embed_dim:   usize,
    pub num_queries: usize,
    pub patch_size:  usize,
}

impl<B: Backend> ReconstructionHead<B> {
    pub fn new(
        input_dim: usize,  // patch_size
        embed_dim: usize,
        num_heads: usize,
        num_queries: usize,
        device: &B::Device,
    ) -> Self {
        let ff_dim = embed_dim * 4;

        Self {
            // Sub-layer 1: self-attention
            self_attn_norm: LunaLayerNorm::new(embed_dim, 1e-5, device),
            self_attn:      FusedMultiheadAttention::new(embed_dim, num_heads, device),
            // Sub-layer 2: cross-attention
            cross_attn_norm: LunaLayerNorm::new(embed_dim, 1e-5, device),
            cross_attn:      FusedMultiheadAttention::new(embed_dim, num_heads, device),
            // Sub-layer 3: FFN
            ffn_norm:    LunaLayerNorm::new(embed_dim, 1e-5, device),
            ffn_linear1: LinearConfig::new(embed_dim, ff_dim).with_bias(true).init(device),
            ffn_linear2: LinearConfig::new(ff_dim, embed_dim).with_bias(true).init(device),
            // Post-decoder
            output_norm: LunaLayerNorm::new(embed_dim, 1e-5, device),
            // Output MLP (no internal norm): fc1(D→4D) → GELU → fc2(4D→patch_size)
            output_fc1: LinearConfig::new(embed_dim, ff_dim).with_bias(true).init(device),
            output_fc2: LinearConfig::new(ff_dim, input_dim).with_bias(true).init(device),
            embed_dim,
            num_queries,
            patch_size: input_dim,
        }
    }

    /// enc:             [B, num_patches, Q*D]
    /// decoder_queries: [B*num_patches, C, D]  (tgt for TransformerDecoder)
    /// num_channels:    C
    /// Returns:         [B, C, T]
    pub fn forward(
        &self,
        enc: Tensor<B, 3>,
        decoder_queries: Tensor<B, 3>,
        num_channels: usize,
    ) -> Tensor<B, 3> {
        let [b, num_patches, _qd] = enc.dims();

        // Reshape encoder output: [B, t, Q*D] → [B*t, Q, D]  (memory for cross-attention)
        let memory = enc.reshape([b * num_patches, self.num_queries, self.embed_dim]);

        // TransformerDecoderLayer with norm_first=True:
        let mut tgt = decoder_queries;

        // Sub-layer 1: self-attention on tgt (norm_first)
        let normed = self.self_attn_norm.forward(tgt.clone());
        let (sa_out, _) = self.self_attn.forward(normed.clone(), normed.clone(), normed);
        tgt = tgt + sa_out;

        // Sub-layer 2: cross-attention tgt→memory (norm_first)
        let normed = self.cross_attn_norm.forward(tgt.clone());
        let (ca_out, _) = self.cross_attn.forward(normed, memory.clone(), memory);
        tgt = tgt + ca_out;

        // Sub-layer 3: FFN (norm_first)
        let normed = self.ffn_norm.forward(tgt.clone());
        let ff_out = self.ffn_linear2.forward(gelu(self.ffn_linear1.forward(normed)));
        let out = tgt + ff_out;

        // Post-decoder norm
        let out = self.output_norm.forward(out);  // [B*t, C, D]

        // Output MLP (no internal norm): fc1 → GELU → fc2
        let out = gelu(self.output_fc1.forward(out));
        let out = self.output_fc2.forward(out);  // [B*t, C, patch_size]

        // Rearrange: [B*t, C, patch_size] → [B, C, t*patch_size]
        // Python: rearrange(out, '(B t) C P -> B C (t P)', B=B)
        out.reshape([b, num_patches, num_channels, self.patch_size])
            .swap_dims(1, 2)  // [B, C, t, patch_size]
            .reshape([b, num_channels, num_patches * self.patch_size])
    }
}
