/// LUNA (Latent Unified Network Architecture) — full model (burn 0.20.1)
///
/// Python: `LUNA` class in models/LUNA.py.
///
/// Architecture:
///   1. PatchEmbedNetwork  — temporal CNN per patch
///   2. FrequencyEmbedder  — FFT + MLP per patch
///   3. NeRF positional encoding of 3D electrode locations
///   4. CrossAttentionBlock — compress C channels → Q queries per time patch
///   5. RotaryTransformerBlocks — temporal encoder on the unified queries
///   6. Head: ReconstructionHead (pretraining) or ClassificationHead (finetune)

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};

use crate::model::patch_embed::PatchEmbedNetwork;
use crate::model::freq_embed::FrequencyFeatureEmbedder;
use crate::model::cross_attention::CrossAttentionBlock;
use crate::model::encoder_block::RotaryEncoderBlock;
use crate::model::norm::LunaLayerNorm;
use crate::model::reconstruction_head::ReconstructionHead;
use crate::model::classification_head::ClassificationHead;
use crate::model::rope::RotaryEmbedding;

/// NeRF-style positional encoding of 3D coordinates.
///
/// Python: `nerf_positional_encoding(coords, embed_size)` in LUNA.py.
/// coords: [N, C, 3] → [N, C, embed_size]
///
/// Python code:
///   freq_bands = 2.0 ** torch.arange(freqs)
///   scaled_coords = coords.unsqueeze(-1) * freq_bands  # \[N,C,dim,freqs\]
///   sin_enc, cos_enc = sin(scaled), cos(scaled)
///   encoded = stack([sin, cos], dim=-1).permute(0,1,3,2,4).reshape(N,C,freqs*dim*2)
///
/// The permute(0,1,3,2,4) swaps dim and freqs axes, so the final flatten
/// order is: for each freq: for each xyz dim: [sin, cos]
pub fn nerf_positional_encoding<B: Backend>(
    coords: Tensor<B, 3>,  // [N, C, 3]
    embed_size: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [n, c, dim] = coords.dims();  // dim = 3
    let freqs = embed_size / (2 * dim);
    let leftover = embed_size - freqs * 2 * dim;

    // freq_bands = 2^(0..freqs)
    let freq_data: Vec<f32> = (0..freqs).map(|i| 2.0_f32.powi(i as i32)).collect();
    let freq_bands = Tensor::<B, 1>::from_data(
        TensorData::new(freq_data, vec![freqs]), device,
    );

    // scaled_coords: [N, C, dim, freqs]
    let coords_4d = coords.unsqueeze_dim::<4>(3);  // [N, C, dim, 1]
    let freq_4d = freq_bands.reshape([1, 1, 1, freqs]);
    let scaled = coords_4d * freq_4d;  // [N, C, dim, freqs]

    let sin_enc = scaled.clone().sin();  // [N, C, dim, freqs]
    let cos_enc = scaled.cos();          // [N, C, dim, freqs]

    // Python: stack([sin, cos], dim=-1) → [N, C, dim, freqs, 2]
    //         .permute(0, 1, 3, 2, 4)  → [N, C, freqs, dim, 2]
    //         .reshape(N, C, freqs * dim * 2)
    let stacked = Tensor::stack::<5>(vec![sin_enc, cos_enc], 4);  // [N, C, dim, freqs, 2]
    let encoded = stacked
        .swap_dims(2, 3)  // [N, C, freqs, dim, 2]
        .reshape([n, c, freqs * dim * 2]);

    if leftover > 0 {
        let pad = Tensor::zeros([n, c, leftover], device);
        Tensor::cat(vec![encoded, pad], 2)
    } else {
        encoded
    }
}

// ── LUNA Model ────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Luna<B: Backend> {
    pub patch_embed: PatchEmbedNetwork<B>,
    pub freq_embed:  FrequencyFeatureEmbedder<B>,
    /// Channel location embedder: timm.Mlp(D, D, 2D, GELU, norm_layer=LayerNorm)
    /// At inference: fc1(D→2D) → GELU → LayerNorm(2D) → fc2(2D→D)
    pub chan_loc_fc1:   Linear<B>,
    pub chan_loc_norm:  LunaLayerNorm<B>,
    pub chan_loc_fc2:   Linear<B>,
    /// Learnable mask token: [1, 1, embed_dim].
    pub mask_token: Param<Tensor<B, 3>>,
    /// Cross-attention channel unification.
    pub cross_attn: CrossAttentionBlock<B>,
    /// Rotary Transformer encoder blocks.
    pub blocks: Vec<RotaryEncoderBlock<B>>,
    /// Final layer norm.
    pub norm: LunaLayerNorm<B>,
    /// Channel name embeddings (for decoder queries in reconstruction).
    pub channel_emb: Option<Embedding<B>>,
    /// Reconstruction head (pretraining).
    pub decoder_head: Option<ReconstructionHead<B>>,
    /// Classification head (finetuning).
    pub classifier: Option<ClassificationHead<B>>,
    // Config
    pub embed_dim:   usize,
    pub num_queries: usize,
    pub patch_size:  usize,
    pub num_heads:   usize,
    pub num_classes: usize,
    pub n_channel_names: usize,
}

impl<B: Backend> Luna<B> {
    pub fn new(
        patch_size:  usize,
        num_queries: usize,
        embed_dim:   usize,
        depth:       usize,
        num_heads:   usize,
        mlp_ratio:   f64,
        norm_eps:    f64,
        num_classes: usize,
        n_channel_names: usize,
        device:      &B::Device,
    ) -> Self {
        let hidden_dim = embed_dim * num_queries;
        let ff_dim = (embed_dim as f64 * mlp_ratio) as usize;

        let patch_embed = PatchEmbedNetwork::new(embed_dim, patch_size, device);
        let freq_embed = FrequencyFeatureEmbedder::new(embed_dim, patch_size, device);

        // Channel location embedder: timm.Mlp(embed_dim, embed_dim, embed_dim*2, GELU, LayerNorm)
        // At inference: fc1(D→2D) → GELU → LayerNorm(2D) → fc2(2D→D)
        let chan_loc_fc1 = LinearConfig::new(embed_dim, embed_dim * 2).with_bias(true).init(device);
        let chan_loc_norm = LunaLayerNorm::new(embed_dim * 2, norm_eps, device);
        let chan_loc_fc2 = LinearConfig::new(embed_dim * 2, embed_dim).with_bias(true).init(device);

        let mask_token = Param::initialized(
            ParamId::new(),
            Tensor::zeros([1, 1, embed_dim], device),
        );

        let cross_attn = CrossAttentionBlock::new(
            num_queries, embed_dim, embed_dim, num_heads, ff_dim, norm_eps, device,
        );

        let total_heads = num_heads * num_queries;
        let blocks = (0..depth)
            .map(|_| RotaryEncoderBlock::new(
                hidden_dim, total_heads, mlp_ratio, true, norm_eps, device,
            ))
            .collect();

        let norm = LunaLayerNorm::new(hidden_dim, norm_eps, device);

        let (channel_emb, decoder_head, classifier) = if num_classes == 0 {
            let emb = EmbeddingConfig::new(n_channel_names, embed_dim).init(device);
            let head = ReconstructionHead::new(
                patch_size, embed_dim, num_heads, num_queries, device,
            );
            (Some(emb), Some(head), None)
        } else {
            let cls = ClassificationHead::new(
                embed_dim, num_queries, num_heads, num_classes, device,
            );
            (None, None, Some(cls))
        };

        Self {
            patch_embed,
            freq_embed,
            chan_loc_fc1, chan_loc_norm, chan_loc_fc2,
            mask_token,
            cross_attn,
            blocks,
            norm,
            channel_emb,
            decoder_head,
            classifier,
            embed_dim,
            num_queries,
            patch_size,
            num_heads,
            num_classes,
            n_channel_names,
        }
    }

    /// Prepare tokens: patch + freq embedding, masking, channel location encoding.
    ///
    /// Mirrors Python `prepare_tokens` exactly.
    ///
    /// Returns: (x_tokenized [B*S, C, D], channel_locations_emb [num_patches*B, C, D])
    ///          where S = num_patches_per_channel = T / patch_size
    fn prepare_tokens(
        &self,
        x_signal: Tensor<B, 3>,
        channel_locations: Tensor<B, 3>,
        mask: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [b, num_channels, t] = x_signal.dims();
        let num_patches = t / self.patch_size;
        let device = x_signal.device();

        // 1. Patch embedding + frequency embedding, then sum
        let x_patched = self.patch_embed.forward(x_signal.clone());  // [B, C*S, D]
        let freq_embed = self.freq_embed.forward(x_signal);          // [B, C*S, D]
        let x_patched = x_patched + freq_embed;

        // 2. Apply mask if provided
        //    Python: mask = rearrange(mask, 'B C (S P) -> B (C S) P', P=patch_size)
        //            mask = (mask.sum(dim=-1) > 0).unsqueeze(-1).float()
        //            x_masked = torch.where(mask.bool(), mask_tokens, x_masked)
        let x_masked = if let Some(ref m) = mask {
            let mask_tokens = self.mask_token.val()
                .expand([b, num_channels * num_patches, self.embed_dim]);
            // [B, C, T] → [B, C*S, P]  (C-first grouping, same as patch_embed)
            let m = m.clone().reshape([b, num_channels * num_patches, self.patch_size]);
            // patch-level mask: [B, C*S, P] → sum over P → [B, C*S, 1] → bool
            let m = m.sum_dim(2);  // [B, C*S, 1] (keepdim)
            let m_bool = m.greater_elem(0.0);  // [B, C*S, 1]
            let m_expanded = m_bool.float()
                .expand([b, num_channels * num_patches, self.embed_dim]);
            // where mask=1 use mask_token, else use patched
            x_patched.clone() * (Tensor::ones_like(&m_expanded) - m_expanded.clone())
                + mask_tokens * m_expanded
        } else {
            x_patched
        };

        // 3. Channel location normalisation to [0,1]
        //    Python: channel_min = min(dim=1, keepdim=True)[0]
        //            channel_max = max(dim=1, keepdim=True)[0]
        //            channel_locations = (loc - min) / (max - min + 1e-8)
        let chan_locs_normed = {
            let mins = channel_locations.clone().min_dim(1);  // [B, 1, 3] keepdim
            let maxs = channel_locations.clone().max_dim(1);  // [B, 1, 3] keepdim
            let range = maxs - mins.clone();
            (channel_locations - mins) / (range + 1e-8)
        };
        // Note: Python adds noise (randn*0.02) when mask is not None — training only, skip

        // 4. NeRF positional encoding → channel_location_embedder (timm.Mlp)
        let chan_locs_encoded = nerf_positional_encoding(
            chan_locs_normed, self.embed_dim, &device,
        );  // [B, C, embed_dim]

        // timm.Mlp at inference: fc1 → GELU → LayerNorm → fc2
        let chan_loc_emb = burn::tensor::activation::gelu(
            self.chan_loc_fc1.forward(chan_locs_encoded)
        );
        let chan_loc_emb = self.chan_loc_norm.forward(chan_loc_emb);
        let chan_loc_emb = self.chan_loc_fc2.forward(chan_loc_emb);  // [B, C, D]

        // 5. Rearrange x_masked for cross-attention
        //    Python: x_tokenized = rearrange(x_masked, 'B (C t) D -> (B t) C D', C=num_channels)
        //    x_masked is [B, C*S, D] where the C*S dim is C-first: [ch0_p0, ch0_p1, ..., ch1_p0, ...]
        //    After rearrange: [B*S, C, D] — for each time-patch, gather all channels
        let x_tokenized = x_masked
            .reshape([b, num_channels, num_patches, self.embed_dim])
            .swap_dims(1, 2)  // [B, S, C, D]
            .reshape([b * num_patches, num_channels, self.embed_dim]);

        // 6. Expand channel location embedding
        //    Python: channel_locations_emb = channel_locations_emb.repeat(num_patches_per_channel, 1, 1)
        //    This tiles the entire [B, C, D] tensor along dim 0, num_patches times:
        //      result shape: [num_patches * B, C, D]
        //      layout: [b0, b1, ..., bB-1, b0, b1, ..., bB-1, ...] repeated num_patches times
        //
        //    BUT the x_tokenized layout after rearrange is: [b0_p0, b0_p1, ..., b0_pS-1, b1_p0, ...]
        //    i.e. [B*S, C, D] grouped by batch then patches.
        //
        //    Looking at the Python more carefully:
        //      x_tokenized = rearrange(x_masked, 'B (C t) D -> (B t) C D', C=num_channels)
        //    einops with C=num_channels splits (C t) as C-major, so t varies fastest:
        //      for B=0: [ch0_p0, ch0_p1, ..., ch1_p0, ...] → [(B=0,t=0), (B=0,t=1), ...]
        //    So x_tokenized dim 0 order is: b0_t0, b0_t1, ..., b0_tS-1, b1_t0, ...
        //
        //    channel_locations_emb.repeat(num_patches, 1, 1) on [B, C, D]:
        //      dim 0 order: b0, b1, ..., bB-1, b0, b1, ..., bB-1 (num_patches copies)
        //
        //    x_tokenized dim 0:   [b0_t0, b0_t1, ..., b0_tS, b1_t0, ...]  (B*S items)
        //    chan_loc_emb dim 0:   [b0, b1, ..., bB, b0, b1, ..., bB, ...] (S*B items, S copies)
        //
        //    For B=1 these are identical. For B>1 the ordering differs at inference
        //    (Python does single-sample inference typically B=1).
        //    But to be safe, match the Python layout exactly:
        //    chan_loc_emb should match x_tokenized: [b0_t0, b0_t1, ..., b1_t0, ...]
        //
        //    Python .repeat(n,1,1) on [B,C,D] gives [n*B, C, D] with B-first tiling.
        //    x_tokenized from einops '(B t)' is B-outer, t-inner.
        //    So for B>1 there IS a mismatch in Python unless B=1!
        //    Looking at pretrain_task_LUNA.py: the model is called with the full batch,
        //    and the channel_locations has shape [B, C, 3]. So Python .repeat(S,1,1)
        //    gives [S*B, C, D] = [b0,b1,...,b0,b1,...] while x_tokenized is [b0_t0,b0_t1,...,b1_t0,...].
        //
        //    This is actually a bug in the Python code when B>1, but it works because
        //    the training code processes it this way and the model learns around it.
        //    For exact parity we replicate the Python behavior.
        let chan_loc_emb = chan_loc_emb.repeat_dim(0, num_patches);
        // Shape: [num_patches * B, C, D] — Python .repeat(num_patches, 1, 1) semantics

        // 7. Add channel location embedding to tokenized signal
        let x_tokenized = x_tokenized + chan_loc_emb.clone();

        (x_tokenized, chan_loc_emb)
    }

    /// Forward pass.
    ///
    /// Returns:
    /// - Classification: (logits [B, num_classes], x_original [B, C, T])
    /// - Reconstruction: (x_reconstructed [B, C, T], x_original [B, C, T], attention_scores [B*S, Q, C])
    pub fn forward(
        &self,
        x_signal: Tensor<B, 3>,
        channel_locations: Tensor<B, 3>,
        mask: Option<Tensor<B, 3>>,
        channel_names: Option<Tensor<B, 2, Int>>,
        rope: &RotaryEmbedding<B>,
    ) -> LunaOutput<B> {
        let x_original = x_signal.clone();
        let [b, num_channels, t] = x_signal.dims();
        let num_patches = t / self.patch_size;

        // 1. Prepare tokens
        let (x_tokenized, chan_loc_emb) = self.prepare_tokens(
            x_signal, channel_locations, mask,
        );

        // 2. Cross-attention: [B*S, C, D] → [B*S, Q, D]
        let (x_unified, attention_scores) = self.cross_attn.forward(x_tokenized);

        // 3. Python: x = rearrange(x, '(B t) Q D -> B t (Q D)', B=B)
        let x = x_unified.reshape([b, num_patches, self.num_queries * self.embed_dim]);

        // 4. Rotary Transformer blocks
        let freqs = rope.get_freqs(num_patches);
        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x, freqs.clone());
        }
        let x_latent = self.norm.forward(x);  // [B, S, Q*D]

        // 5. Head
        if let Some(ref classifier) = self.classifier {
            let logits = classifier.forward(x_latent);  // [B, num_classes]
            LunaOutput::Classification {
                logits,
                x_original,
            }
        } else if let Some(ref decoder_head) = self.decoder_head {
            // Build decoder queries: channel_locations_emb + channel_name_emb
            let mut decoder_queries = chan_loc_emb;  // [num_patches*B, C, D]
            if let (Some(ref emb), Some(names)) = (&self.channel_emb, channel_names) {
                // Python: channel_emb = self.channel_emb(channel_names)  # [B, C, D]
                //         channel_emb = channel_emb.repeat(num_patches, 1, 1)
                let ch_emb = emb.forward(names);  // [B, C, D]
                let ch_emb = ch_emb.repeat_dim(0, num_patches);  // [num_patches*B, C, D]
                decoder_queries = decoder_queries + ch_emb;
            }

            let x_reconstructed = decoder_head.forward(
                x_latent, decoder_queries, num_channels,
            );
            LunaOutput::Reconstruction {
                x_reconstructed,
                x_original,
                attention_scores,
            }
        } else {
            panic!("LUNA model has neither classifier nor decoder head");
        }
    }
}

/// Output enum to match Python's variable return signature.
pub enum LunaOutput<B: Backend> {
    /// Classification: (logits, x_original)
    Classification {
        logits: Tensor<B, 3>,
        x_original: Tensor<B, 3>,
    },
    /// Reconstruction: (x_reconstructed, x_original, attention_scores)
    Reconstruction {
        x_reconstructed: Tensor<B, 3>,
        x_original: Tensor<B, 3>,
        attention_scores: Tensor<B, 3>,
    },
}
