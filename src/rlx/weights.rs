//! Safetensors → flat parameter map for RLX graphs.

use std::collections::HashMap;

use half::bf16;
use safetensors::SafeTensors;

use crate::config::ModelConfig;

#[derive(Clone, Debug)]
pub struct ParamBuf {
    pub data:  Vec<f32>,
    pub shape: Vec<usize>,
}

pub type ParamMap = HashMap<String, ParamBuf>;

pub fn load_safetensors(path: &str) -> anyhow::Result<HashMap<String, ParamBuf>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::with_capacity(st.len());
    for (raw_key, view) in st.tensors() {
        let key = raw_key
            .strip_prefix("model.")
            .unwrap_or(raw_key.as_str())
            .to_string();
        let shape: Vec<usize> = view.shape().to_vec();
        let data = match view.dtype() {
            safetensors::Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => anyhow::bail!("unsupported dtype {:?} for key {}", other, key),
        };
        out.insert(key, ParamBuf { data, shape });
    }
    Ok(out)
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn take_linear_w(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    let p = raw
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))?;
    anyhow::ensure!(
        p.shape.len() == 2,
        "Linear weight {key} must be 2-D, got {:?}",
        p.shape
    );
    let (out_d, in_d) = (p.shape[0], p.shape[1]);
    let data = transpose(&p.data, out_d, in_d);
    Ok(ParamBuf {
        data,
        shape: vec![in_d, out_d],
    })
}

fn take(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    raw.remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))
}

/// Split a fused `in_proj` weight `[dim, 3*dim]` into Q/K/V `[dim, dim]`.
fn split_in_proj(p: ParamBuf, dim: usize) -> anyhow::Result<(ParamBuf, ParamBuf, ParamBuf)> {
    anyhow::ensure!(
        p.shape == vec![dim, 3 * dim],
        "in_proj shape mismatch: {:?}",
        p.shape
    );
    let mut wq = vec![0f32; dim * dim];
    let mut wk = vec![0f32; dim * dim];
    let mut wv = vec![0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let src = i * 3 * dim + j;
            wq[i * dim + j] = p.data[src];
            wk[i * dim + j] = p.data[src + dim];
            wv[i * dim + j] = p.data[src + 2 * dim];
        }
    }
    Ok((
        ParamBuf {
            data: wq,
            shape: vec![dim, dim],
        },
        ParamBuf {
            data: wk,
            shape: vec![dim, dim],
        },
        ParamBuf {
            data: wv,
            shape: vec![dim, dim],
        },
    ))
}

fn split_in_proj_bias(p: ParamBuf, dim: usize) -> anyhow::Result<(ParamBuf, ParamBuf, ParamBuf)> {
    anyhow::ensure!(p.shape == vec![3 * dim], "in_proj bias shape mismatch");
    Ok((
        ParamBuf {
            data: p.data[0..dim].to_vec(),
            shape: vec![dim],
        },
        ParamBuf {
            data: p.data[dim..2 * dim].to_vec(),
            shape: vec![dim],
        },
        ParamBuf {
            data: p.data[2 * dim..3 * dim].to_vec(),
            shape: vec![dim],
        },
    ))
}

fn insert_fused_mha(
    p: &mut ParamMap,
    prefix: &str,
    raw: &mut HashMap<String, ParamBuf>,
    dim: usize,
) -> anyhow::Result<()> {
    let w = take_linear_w(raw, &format!("{prefix}.in_proj_weight"))?;
    let (wq, wk, wv) = split_in_proj(w, dim)?;
    p.insert(format!("{prefix}.wq.weight"), wq);
    p.insert(format!("{prefix}.wk.weight"), wk);
    p.insert(format!("{prefix}.wv.weight"), wv);
    if let Ok(b) = take(raw, &format!("{prefix}.in_proj_bias")) {
        let (bq, bk, bv) = split_in_proj_bias(b, dim)?;
        p.insert(format!("{prefix}.wq.bias"), bq);
        p.insert(format!("{prefix}.wk.bias"), bk);
        p.insert(format!("{prefix}.wv.bias"), bv);
    }
    p.insert(
        format!("{prefix}.wo.weight"),
        take_linear_w(raw, &format!("{prefix}.out_proj.weight"))?,
    );
    if let Ok(b) = take(raw, &format!("{prefix}.out_proj.bias")) {
        p.insert(format!("{prefix}.wo.bias"), b);
    }
    Ok(())
}

/// Build the parameter map consumed by [`super::graph::build_forward_graph`].
pub fn build_forward_params(
    raw: &mut HashMap<String, ParamBuf>,
    cfg: &ModelConfig,
) -> anyhow::Result<ParamMap> {
    let d = cfg.embed_dim;
    let hidden = cfg.hidden_dim();
    let ff_ca = (d as f64 * cfg.mlp_ratio) as usize;
    let ff_rot = cfg.ffn_hidden_dim();
    let mut p = ParamMap::new();

    // ── Cross-attention block ───────────────────────────────────────────────
    p.insert(
        "cross_attn.query_embed".into(),
        take(raw, "cross_attn.query_embed")?,
    );
    for norm in ["queries_norm", "keys_norm", "values_norm"] {
        p.insert(
            format!("cross_attn.{norm}.weight"),
            take(raw, &format!("cross_attn.{norm}.weight"))?,
        );
        p.insert(
            format!("cross_attn.{norm}.bias"),
            take(raw, &format!("cross_attn.{norm}.bias"))?,
        );
    }
    insert_fused_mha(&mut p, "cross_attn.cross_attention", raw, d)?;
    p.insert(
        "cross_attn.ffn.fc1.weight".into(),
        take_linear_w(raw, "cross_attn.ffn.fc1.weight")?,
    );
    p.insert(
        "cross_attn.ffn.fc1.bias".into(),
        take(raw, "cross_attn.ffn.fc1.bias")?,
    );
    p.insert(
        "cross_attn.ffn.norm.weight".into(),
        take(raw, "cross_attn.ffn.norm.weight")?,
    );
    p.insert(
        "cross_attn.ffn.norm.bias".into(),
        take(raw, "cross_attn.ffn.norm.bias")?,
    );
    p.insert(
        "cross_attn.ffn.fc2.weight".into(),
        take_linear_w(raw, "cross_attn.ffn.fc2.weight")?,
    );
    p.insert(
        "cross_attn.ffn.fc2.bias".into(),
        take(raw, "cross_attn.ffn.fc2.bias")?,
    );
    for i in 0..3 {
        let q = format!("cross_attn.query_self_attn.layers.{i}");
        p.insert(
            format!("{q}.norm1.weight"),
            take(raw, &format!("{q}.norm1.weight"))?,
        );
        p.insert(
            format!("{q}.norm1.bias"),
            take(raw, &format!("{q}.norm1.bias"))?,
        );
        insert_fused_mha(&mut p, &format!("{q}.self_attn"), raw, d)?;
        p.insert(
            format!("{q}.norm2.weight"),
            take(raw, &format!("{q}.norm2.weight"))?,
        );
        p.insert(
            format!("{q}.norm2.bias"),
            take(raw, &format!("{q}.norm2.bias"))?,
        );
        p.insert(
            format!("{q}.linear1.weight"),
            take_linear_w(raw, &format!("{q}.linear1.weight"))?,
        );
        p.insert(
            format!("{q}.linear1.bias"),
            take(raw, &format!("{q}.linear1.bias"))?,
        );
        p.insert(
            format!("{q}.linear2.weight"),
            take_linear_w(raw, &format!("{q}.linear2.weight"))?,
        );
        p.insert(
            format!("{q}.linear2.bias"),
            take(raw, &format!("{q}.linear2.bias"))?,
        );
    }

    // ── Rotary transformer blocks ───────────────────────────────────────────
    for i in 0..cfg.depth {
        let q = format!("blocks.{i}");
        p.insert(
            format!("{q}.norm1.weight"),
            take(raw, &format!("{q}.norm1.weight"))?,
        );
        p.insert(
            format!("{q}.norm1.bias"),
            take(raw, &format!("{q}.norm1.bias"))?,
        );
        let qkv = take_linear_w(raw, &format!("{q}.attn.qkv_proj.weight"))?;
        let (wq, wk, wv) = split_in_proj(qkv, hidden)?;
        p.insert(format!("{q}.attn.wq.weight"), wq);
        p.insert(format!("{q}.attn.wk.weight"), wk);
        p.insert(format!("{q}.attn.wv.weight"), wv);
        if let Ok(b) = take(raw, &format!("{q}.attn.qkv_proj.bias")) {
            let (bq, bk, bv) = split_in_proj_bias(b, hidden)?;
            p.insert(format!("{q}.attn.wq.bias"), bq);
            p.insert(format!("{q}.attn.wk.bias"), bk);
            p.insert(format!("{q}.attn.wv.bias"), bv);
        }
        p.insert(
            format!("{q}.attn.proj.weight"),
            take_linear_w(raw, &format!("{q}.attn.proj.weight"))?,
        );
        p.insert(
            format!("{q}.attn.proj.bias"),
            take(raw, &format!("{q}.attn.proj.bias"))?,
        );
        p.insert(
            format!("{q}.norm2.weight"),
            take(raw, &format!("{q}.norm2.weight"))?,
        );
        p.insert(
            format!("{q}.norm2.bias"),
            take(raw, &format!("{q}.norm2.bias"))?,
        );
        p.insert(
            format!("{q}.mlp.fc1.weight"),
            take_linear_w(raw, &format!("{q}.mlp.fc1.weight"))?,
        );
        p.insert(
            format!("{q}.mlp.fc1.bias"),
            take(raw, &format!("{q}.mlp.fc1.bias"))?,
        );
        p.insert(
            format!("{q}.mlp.norm.weight"),
            take(raw, &format!("{q}.mlp.norm.weight"))?,
        );
        p.insert(
            format!("{q}.mlp.norm.bias"),
            take(raw, &format!("{q}.mlp.norm.bias"))?,
        );
        p.insert(
            format!("{q}.mlp.fc2.weight"),
            take_linear_w(raw, &format!("{q}.mlp.fc2.weight"))?,
        );
        p.insert(
            format!("{q}.mlp.fc2.bias"),
            take(raw, &format!("{q}.mlp.fc2.bias"))?,
        );
    }

    p.insert("norm.weight".into(), take(raw, "norm.weight")?);
    p.insert("norm.bias".into(), take(raw, "norm.bias")?);

    if cfg.num_classes > 0 {
        if let Ok(agg) = take(raw, "classifier.learned_agg") {
            p.insert("classifier.learned_agg".into(), agg);
        }
        insert_fused_mha(&mut p, "classifier.decoder_attn", raw, cfg.hidden_dim())?;
        p.insert(
            "classifier.decoder_ffn.fc1.weight".into(),
            take_linear_w(raw, "classifier.decoder_ffn.fc1.weight")?,
        );
        p.insert(
            "classifier.decoder_ffn.fc1.bias".into(),
            take(raw, "classifier.decoder_ffn.fc1.bias")?,
        );
        p.insert(
            "classifier.decoder_ffn.fc2.weight".into(),
            take_linear_w(raw, "classifier.decoder_ffn.fc2.weight")?,
        );
        p.insert(
            "classifier.decoder_ffn.fc2.bias".into(),
            take(raw, "classifier.decoder_ffn.fc2.bias")?,
        );
        return Ok(p);
    }

    // ── Reconstruction head ─────────────────────────────────────────────────
    let dp = "decoder_head.decoder_pred.layers.0";
    p.insert(
        format!("{dp}.norm1.weight"),
        take(raw, &format!("{dp}.norm1.weight"))?,
    );
    p.insert(
        format!("{dp}.norm1.bias"),
        take(raw, &format!("{dp}.norm1.bias"))?,
    );
    insert_fused_mha(&mut p, &format!("{dp}.self_attn"), raw, d)?;
    p.insert(
        format!("{dp}.norm2.weight"),
        take(raw, &format!("{dp}.norm2.weight"))?,
    );
    p.insert(
        format!("{dp}.norm2.bias"),
        take(raw, &format!("{dp}.norm2.bias"))?,
    );
    insert_fused_mha(&mut p, &format!("{dp}.multihead_attn"), raw, d)?;
    p.insert(
        format!("{dp}.norm3.weight"),
        take(raw, &format!("{dp}.norm3.weight"))?,
    );
    p.insert(
        format!("{dp}.norm3.bias"),
        take(raw, &format!("{dp}.norm3.bias"))?,
    );
    p.insert(
        format!("{dp}.linear1.weight"),
        take_linear_w(raw, &format!("{dp}.linear1.weight"))?,
    );
    p.insert(
        format!("{dp}.linear1.bias"),
        take(raw, &format!("{dp}.linear1.bias"))?,
    );
    p.insert(
        format!("{dp}.linear2.weight"),
        take_linear_w(raw, &format!("{dp}.linear2.weight"))?,
    );
    p.insert(
        format!("{dp}.linear2.bias"),
        take(raw, &format!("{dp}.linear2.bias"))?,
    );
    p.insert(
        "decoder_head.norm.weight".into(),
        take(raw, "decoder_head.norm.weight")?,
    );
    p.insert(
        "decoder_head.norm.bias".into(),
        take(raw, "decoder_head.norm.bias")?,
    );
    p.insert(
        "decoder_head.decoder_linear.fc1.weight".into(),
        take_linear_w(raw, "decoder_head.decoder_linear.fc1.weight")?,
    );
    p.insert(
        "decoder_head.decoder_linear.fc1.bias".into(),
        take(raw, "decoder_head.decoder_linear.fc1.bias")?,
    );
    p.insert(
        "decoder_head.decoder_linear.fc2.weight".into(),
        take_linear_w(raw, "decoder_head.decoder_linear.fc2.weight")?,
    );
    p.insert(
        "decoder_head.decoder_linear.fc2.bias".into(),
        take(raw, "decoder_head.decoder_linear.fc2.bias")?,
    );

    let _ = (ff_ca, ff_rot);
    Ok(p)
}

/// Parameters for the CPU token-preparation path (patch / freq / channel MLP).
pub fn build_prepare_params(
    raw: &mut HashMap<String, ParamBuf>,
) -> anyhow::Result<ParamMap> {
    let mut p = ParamMap::new();
    // proj_in layout: conv0, gn1, gn2, conv3, gn4, gn5, conv6, gn7, …
    for (conv_idx, gn_idx, conv, gn) in [(0, 1, "conv1", 1), (3, 4, "conv2", 2), (6, 7, "conv3", 3)] {
        p.insert(
            format!("patch_embed.{conv}.weight"),
            take(raw, &format!("patch_embed.proj_in.{conv_idx}.weight"))?,
        );
        p.insert(
            format!("patch_embed.{conv}.bias"),
            take(raw, &format!("patch_embed.proj_in.{conv_idx}.bias"))?,
        );
        p.insert(
            format!("patch_embed.gn{gn}.weight"),
            take(raw, &format!("patch_embed.proj_in.{gn_idx}.weight"))?,
        );
        p.insert(
            format!("patch_embed.gn{gn}.bias"),
            take(raw, &format!("patch_embed.proj_in.{gn_idx}.bias"))?,
        );
    }
    p.insert(
        "freq_embed.fc1.weight".into(),
        take_linear_w(raw, "freq_embed.frequency_to_embed.fc1.weight")?,
    );
    p.insert(
        "freq_embed.fc1.bias".into(),
        take(raw, "freq_embed.frequency_to_embed.fc1.bias")?,
    );
    p.insert(
        "freq_embed.fc2.weight".into(),
        take_linear_w(raw, "freq_embed.frequency_to_embed.fc2.weight")?,
    );
    p.insert(
        "freq_embed.fc2.bias".into(),
        take(raw, "freq_embed.frequency_to_embed.fc2.bias")?,
    );
    p.insert(
        "chan_loc.fc1.weight".into(),
        take_linear_w(raw, "channel_location_embedder.0.fc1.weight")?,
    );
    p.insert(
        "chan_loc.fc1.bias".into(),
        take(raw, "channel_location_embedder.0.fc1.bias")?,
    );
    p.insert(
        "chan_loc.norm.weight".into(),
        take(raw, "channel_location_embedder.0.norm.weight")?,
    );
    p.insert(
        "chan_loc.norm.bias".into(),
        take(raw, "channel_location_embedder.0.norm.bias")?,
    );
    p.insert(
        "chan_loc.fc2.weight".into(),
        take_linear_w(raw, "channel_location_embedder.0.fc2.weight")?,
    );
    p.insert(
        "chan_loc.fc2.bias".into(),
        take(raw, "channel_location_embedder.0.fc2.bias")?,
    );
    if let Ok(t) = take(raw, "channel_emb.embeddings.weight") {
        // Embedding table [vocab, D] — keep as-is for gather on CPU.
        p.insert("channel_emb.weight".into(), t);
    }
    Ok(p)
}

pub fn apply_params(compiled: &mut rlx::CompiledGraph, params: &ParamMap) {
    for (name, buf) in params {
        compiled.set_param(name, &buf.data);
    }
}
