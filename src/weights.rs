/// Load pretrained LUNA weights from a safetensors file (burn 0.20.1)
///
/// Weights are stored as float32 or bfloat16.  Keys match PyTorch Lightning
/// state_dict naming with the `model.` prefix stripped.
///
/// Key patterns for fused nn.MultiheadAttention:
///   <prefix>.in_proj_weight  [3*D, D]
///   <prefix>.in_proj_bias    [3*D]
///   <prefix>.out_proj.weight [D, D]
///   <prefix>.out_proj.bias   [D]
///
/// Key patterns for nn.TransformerEncoderLayer (norm_first):
///   <prefix>.self_attn.in_proj_weight  [3*D, D]
///   <prefix>.self_attn.in_proj_bias    [3*D]
///   <prefix>.self_attn.out_proj.weight [D, D]
///   <prefix>.self_attn.out_proj.bias   [D]
///   <prefix>.norm1.weight [D]
///   <prefix>.norm1.bias   [D]
///   <prefix>.norm2.weight [D]
///   <prefix>.norm2.bias   [D]
///   <prefix>.linear1.weight [ff_dim, D]
///   <prefix>.linear1.bias   [ff_dim]
///   <prefix>.linear2.weight [D, ff_dim]
///   <prefix>.linear2.bias   [D]

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;

use crate::model::luna::Luna;
use crate::config::ModelConfig;

// ── Raw tensor map ────────────────────────────────────────────────────────────

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::new();

        for (raw_key, view) in st.tensors() {
            let key = raw_key
                .strip_prefix("model.")
                .unwrap_or(raw_key.as_str())
                .to_string();

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    pub fn get<B: Backend, const N: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.get(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;
        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }
        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data.clone(), shape.clone()),
            device,
        ))
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

/// PyTorch [out, in] → burn [in, out]
#[allow(dead_code)]
fn set_linear_w<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
}

fn set_linear_wb<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = linear.bias {
        linear.bias = Some(bias.clone().map(|_| b));
    }
}

fn set_layernorm<B: Backend>(norm: &mut crate::model::norm::LunaLayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    norm.inner.gamma = norm.inner.gamma.clone().map(|_| w);
    if let Some(ref beta) = norm.inner.beta {
        norm.inner.beta = Some(beta.clone().map(|_| b));
    }
}

fn set_groupnorm<B: Backend>(gn: &mut burn::nn::GroupNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    if let Some(ref gamma) = gn.gamma {
        gn.gamma = Some(gamma.clone().map(|_| w));
    }
    if let Some(ref beta) = gn.beta {
        gn.beta = Some(beta.clone().map(|_| b));
    }
}

fn set_conv2d_wb<B: Backend>(conv: &mut burn::nn::conv::Conv2d<B>, w: Tensor<B, 4>, b: Tensor<B, 1>) {
    conv.weight = conv.weight.clone().map(|_| w);
    if let Some(ref bias) = conv.bias {
        conv.bias = Some(bias.clone().map(|_| b));
    }
}

/// Load nn.MultiheadAttention fused weights into FusedMultiheadAttention.
/// PyTorch stores in_proj_weight [3*D, D] and in_proj_bias [3*D].
/// Our FusedMultiheadAttention has a single Linear(D, 3*D).
fn load_fused_mha<B: Backend>(
    wm: &WeightMap,
    mha: &mut crate::model::cross_attention::FusedMultiheadAttention<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>(&format!("{prefix}.in_proj_weight"), device),
        wm.get::<B, 1>(&format!("{prefix}.in_proj_bias"), device),
    ) {
        set_linear_wb(&mut mha.in_proj, w, b);
    }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>(&format!("{prefix}.out_proj.weight"), device),
        wm.get::<B, 1>(&format!("{prefix}.out_proj.bias"), device),
    ) {
        set_linear_wb(&mut mha.out_proj, w, b);
    }
    Ok(())
}

/// Load nn.TransformerEncoderLayer weights.
fn load_encoder_layer<B: Backend>(
    wm: &WeightMap,
    layer: &mut crate::model::cross_attention::TransformerEncoderLayer<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    // norm1
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>(&format!("{prefix}.norm1.weight"), device),
        wm.get::<B, 1>(&format!("{prefix}.norm1.bias"), device),
    ) {
        set_layernorm(&mut layer.norm1, w, b);
    }
    // self_attn
    load_fused_mha(wm, &mut layer.self_attn, &format!("{prefix}.self_attn"), device)?;
    // norm2
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>(&format!("{prefix}.norm2.weight"), device),
        wm.get::<B, 1>(&format!("{prefix}.norm2.bias"), device),
    ) {
        set_layernorm(&mut layer.norm2, w, b);
    }
    // linear1
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>(&format!("{prefix}.linear1.weight"), device),
        wm.get::<B, 1>(&format!("{prefix}.linear1.bias"), device),
    ) {
        set_linear_wb(&mut layer.linear1, w, b);
    }
    // linear2
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>(&format!("{prefix}.linear2.weight"), device),
        wm.get::<B, 1>(&format!("{prefix}.linear2.bias"), device),
    ) {
        set_linear_wb(&mut layer.linear2, w, b);
    }
    Ok(())
}

// ── Full model loader ─────────────────────────────────────────────────────────

/// Load a LUNA model from a safetensors file.
pub fn load_model<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    n_channel_names: usize,
    device: &B::Device,
) -> anyhow::Result<Luna<B>> {
    let wm = WeightMap::from_file(weights_path)?;
    eprintln!("Loading {} weight tensors...", wm.tensors.len());
    load_model_from_wm(cfg, &wm, n_channel_names, device)
}

/// Load a LUNA model from a pre-loaded [`WeightMap`].
///
/// Useful for loading from quantized/dequantized weights.
pub fn load_model_from_wm<B: Backend>(
    cfg: &ModelConfig,
    wm: &WeightMap,
    n_channel_names: usize,
    device: &B::Device,
) -> anyhow::Result<Luna<B>> {
    let mut model = Luna::new(
        cfg.patch_size, cfg.num_queries, cfg.embed_dim, cfg.depth,
        cfg.num_heads, cfg.mlp_ratio, cfg.norm_eps, cfg.num_classes,
        n_channel_names, device,
    );

    load_luna_weights(wm, &mut model, device)?;
    Ok(model)
}

fn load_luna_weights<B: Backend>(
    wm: &WeightMap,
    model: &mut Luna<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    // ── Patch embedding network (nn.Sequential indexed) ─────────────────────
    // proj_in.0 = Conv2d, proj_in.1 = GroupNorm, proj_in.2 = GELU (no params)
    // proj_in.3 = Conv2d, proj_in.4 = GroupNorm, proj_in.5 = GELU
    // proj_in.6 = Conv2d, proj_in.7 = GroupNorm, proj_in.8 = GELU
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 4>("patch_embed.proj_in.0.weight", device),
        wm.get::<B, 1>("patch_embed.proj_in.0.bias", device),
    ) { set_conv2d_wb(&mut model.patch_embed.conv1, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("patch_embed.proj_in.1.weight", device),
        wm.get::<B, 1>("patch_embed.proj_in.1.bias", device),
    ) { set_groupnorm(&mut model.patch_embed.gn1, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 4>("patch_embed.proj_in.3.weight", device),
        wm.get::<B, 1>("patch_embed.proj_in.3.bias", device),
    ) { set_conv2d_wb(&mut model.patch_embed.conv2, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("patch_embed.proj_in.4.weight", device),
        wm.get::<B, 1>("patch_embed.proj_in.4.bias", device),
    ) { set_groupnorm(&mut model.patch_embed.gn2, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 4>("patch_embed.proj_in.6.weight", device),
        wm.get::<B, 1>("patch_embed.proj_in.6.bias", device),
    ) { set_conv2d_wb(&mut model.patch_embed.conv3, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("patch_embed.proj_in.7.weight", device),
        wm.get::<B, 1>("patch_embed.proj_in.7.bias", device),
    ) { set_groupnorm(&mut model.patch_embed.gn3, w, b); }

    // ── Frequency embedder ──────────────────────────────────────────────────
    // Python: freq_embed.frequency_to_embed = timm.Mlp(fc1, fc2)
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>("freq_embed.frequency_to_embed.fc1.weight", device),
        wm.get::<B, 1>("freq_embed.frequency_to_embed.fc1.bias", device),
    ) { set_linear_wb(&mut model.freq_embed.fc1, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>("freq_embed.frequency_to_embed.fc2.weight", device),
        wm.get::<B, 1>("freq_embed.frequency_to_embed.fc2.bias", device),
    ) { set_linear_wb(&mut model.freq_embed.fc2, w, b); }

    // ── Channel location embedder ───────────────────────────────────────────
    // Python: channel_location_embedder = nn.Sequential(Mlp(...))
    // The Mlp is at index 0 in the Sequential, so keys are:
    //   channel_location_embedder.0.fc1.weight/bias
    //   channel_location_embedder.0.norm.weight/bias  (LayerNorm between fc1 and fc2)
    //   channel_location_embedder.0.fc2.weight/bias
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>("channel_location_embedder.0.fc1.weight", device),
        wm.get::<B, 1>("channel_location_embedder.0.fc1.bias", device),
    ) { set_linear_wb(&mut model.chan_loc_fc1, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("channel_location_embedder.0.norm.weight", device),
        wm.get::<B, 1>("channel_location_embedder.0.norm.bias", device),
    ) { set_layernorm(&mut model.chan_loc_norm, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>("channel_location_embedder.0.fc2.weight", device),
        wm.get::<B, 1>("channel_location_embedder.0.fc2.bias", device),
    ) { set_linear_wb(&mut model.chan_loc_fc2, w, b); }

    // ── Mask token ──────────────────────────────────────────────────────────
    if let Ok(t) = wm.get::<B, 3>("mask_token", device) {
        model.mask_token = model.mask_token.clone().map(|_| t);
    }

    // ── Cross-attention block ───────────────────────────────────────────────
    if let Ok(t) = wm.get::<B, 3>("cross_attn.query_embed", device) {
        model.cross_attn.query_embed = model.cross_attn.query_embed.clone().map(|_| t);
    }
    if wm.has("cross_attn.temparature") {
        if let Ok(t) = wm.get::<B, 1>("cross_attn.temparature", device) {
            model.cross_attn.temperature = model.cross_attn.temperature.clone().map(|_| t);
        }
    }
    // Cross-attention (nn.MultiheadAttention)
    load_fused_mha(wm, &mut model.cross_attn.cross_attention, "cross_attn.cross_attention", device)?;
    // FFN (timm.Mlp with norm_layer)
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>("cross_attn.ffn.fc1.weight", device),
        wm.get::<B, 1>("cross_attn.ffn.fc1.bias", device),
    ) { set_linear_wb(&mut model.cross_attn.ffn_fc1, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("cross_attn.ffn.norm.weight", device),
        wm.get::<B, 1>("cross_attn.ffn.norm.bias", device),
    ) { set_layernorm(&mut model.cross_attn.ffn_norm, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 2>("cross_attn.ffn.fc2.weight", device),
        wm.get::<B, 1>("cross_attn.ffn.fc2.bias", device),
    ) { set_linear_wb(&mut model.cross_attn.ffn_fc2, w, b); }
    // Pre-attention norms
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("cross_attn.queries_norm.weight", device),
        wm.get::<B, 1>("cross_attn.queries_norm.bias", device),
    ) { set_layernorm(&mut model.cross_attn.queries_norm, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("cross_attn.keys_norm.weight", device),
        wm.get::<B, 1>("cross_attn.keys_norm.bias", device),
    ) { set_layernorm(&mut model.cross_attn.keys_norm, w, b); }
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("cross_attn.values_norm.weight", device),
        wm.get::<B, 1>("cross_attn.values_norm.bias", device),
    ) { set_layernorm(&mut model.cross_attn.values_norm, w, b); }
    // Query self-attention (3-layer TransformerEncoder)
    // Python: cross_attn.query_self_attn.layers.{i}.<sublayer>
    for (i, layer) in model.cross_attn.self_attn_layers.iter_mut().enumerate() {
        let p = format!("cross_attn.query_self_attn.layers.{i}");
        load_encoder_layer(wm, layer, &p, device)?;
    }

    // ── Rotary Transformer blocks ───────────────────────────────────────────
    // Python: blocks.{i}.norm1, blocks.{i}.attn (RotarySelfAttentionBlock), blocks.{i}.norm2, blocks.{i}.mlp
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("blocks.{i}");
        // norm1
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>(&format!("{p}.norm1.weight"), device),
            wm.get::<B, 1>(&format!("{p}.norm1.bias"), device),
        ) { set_layernorm(&mut block.norm1, w, b); }
        // attn: RotarySelfAttentionBlock has qkv_proj (Linear) and proj (Linear)
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>(&format!("{p}.attn.qkv_proj.weight"), device),
            wm.get::<B, 1>(&format!("{p}.attn.qkv_proj.bias"), device),
        ) { set_linear_wb(&mut block.attn.qkv, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>(&format!("{p}.attn.proj.weight"), device),
            wm.get::<B, 1>(&format!("{p}.attn.proj.bias"), device),
        ) { set_linear_wb(&mut block.attn.proj, w, b); }
        // norm2
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>(&format!("{p}.norm2.weight"), device),
            wm.get::<B, 1>(&format!("{p}.norm2.bias"), device),
        ) { set_layernorm(&mut block.norm2, w, b); }
        // mlp: FeedForwardBlock has fc1, norm (LayerNorm on hidden), fc2
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>(&format!("{p}.mlp.fc1.weight"), device),
            wm.get::<B, 1>(&format!("{p}.mlp.fc1.bias"), device),
        ) { set_linear_wb(&mut block.mlp.fc1, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>(&format!("{p}.mlp.norm.weight"), device),
            wm.get::<B, 1>(&format!("{p}.mlp.norm.bias"), device),
        ) { set_layernorm(&mut block.mlp.norm, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>(&format!("{p}.mlp.fc2.weight"), device),
            wm.get::<B, 1>(&format!("{p}.mlp.fc2.bias"), device),
        ) { set_linear_wb(&mut block.mlp.fc2, w, b); }
    }

    // ── Final norm ──────────────────────────────────────────────────────────
    if let (Ok(w), Ok(b)) = (
        wm.get::<B, 1>("norm.weight", device),
        wm.get::<B, 1>("norm.bias", device),
    ) { set_layernorm(&mut model.norm, w, b); }

    // ── Reconstruction head (decoder_head) ──────────────────────────────────
    if let Some(ref mut head) = model.decoder_head {
        // TransformerDecoderLayer at index 0 in the TransformerDecoder
        let dp = "decoder_head.decoder_pred.layers.0";
        // Sub-layer 1: self_attn
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>(&format!("{dp}.norm1.weight"), device),
            wm.get::<B, 1>(&format!("{dp}.norm1.bias"), device),
        ) { set_layernorm(&mut head.self_attn_norm, w, b); }
        load_fused_mha(wm, &mut head.self_attn, &format!("{dp}.self_attn"), device)?;
        // Sub-layer 2: multihead_attn (cross-attention)
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>(&format!("{dp}.norm2.weight"), device),
            wm.get::<B, 1>(&format!("{dp}.norm2.bias"), device),
        ) { set_layernorm(&mut head.cross_attn_norm, w, b); }
        load_fused_mha(wm, &mut head.cross_attn, &format!("{dp}.multihead_attn"), device)?;
        // Sub-layer 3: FFN
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>(&format!("{dp}.norm3.weight"), device),
            wm.get::<B, 1>(&format!("{dp}.norm3.bias"), device),
        ) { set_layernorm(&mut head.ffn_norm, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>(&format!("{dp}.linear1.weight"), device),
            wm.get::<B, 1>(&format!("{dp}.linear1.bias"), device),
        ) { set_linear_wb(&mut head.ffn_linear1, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>(&format!("{dp}.linear2.weight"), device),
            wm.get::<B, 1>(&format!("{dp}.linear2.bias"), device),
        ) { set_linear_wb(&mut head.ffn_linear2, w, b); }
        // Post-decoder norm
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 1>("decoder_head.norm.weight", device),
            wm.get::<B, 1>("decoder_head.norm.bias", device),
        ) { set_layernorm(&mut head.output_norm, w, b); }
        // Output MLP: decoder_head.decoder_linear = timm.Mlp(fc1, fc2)
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>("decoder_head.decoder_linear.fc1.weight", device),
            wm.get::<B, 1>("decoder_head.decoder_linear.fc1.bias", device),
        ) { set_linear_wb(&mut head.output_fc1, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>("decoder_head.decoder_linear.fc2.weight", device),
            wm.get::<B, 1>("decoder_head.decoder_linear.fc2.bias", device),
        ) { set_linear_wb(&mut head.output_fc2, w, b); }
    }

    // ── Classification head ─────────────────────────────────────────────────
    if let Some(ref mut cls) = model.classifier {
        if let Ok(t) = wm.get::<B, 3>("classifier.learned_agg", device) {
            cls.learned_agg = cls.learned_agg.clone().map(|_| t);
        }
        load_fused_mha(wm, &mut cls.decoder_attn, "classifier.decoder_attn", device)?;
        // FFN: classifier.decoder_ffn = timm.Mlp(fc1, fc2)
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>("classifier.decoder_ffn.fc1.weight", device),
            wm.get::<B, 1>("classifier.decoder_ffn.fc1.bias", device),
        ) { set_linear_wb(&mut cls.ffn_fc1, w, b); }
        if let (Ok(w), Ok(b)) = (
            wm.get::<B, 2>("classifier.decoder_ffn.fc2.weight", device),
            wm.get::<B, 1>("classifier.decoder_ffn.fc2.bias", device),
        ) { set_linear_wb(&mut cls.ffn_fc2, w, b); }
    }

    // ── Channel embeddings ──────────────────────────────────────────────────
    if let Some(ref mut emb) = model.channel_emb {
        if let Ok(w) = wm.get::<B, 2>("channel_emb.embeddings.weight", device) {
            emb.weight = emb.weight.clone().map(|_| w);
        }
    }

    Ok(())
}
