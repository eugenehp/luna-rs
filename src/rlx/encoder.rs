//! RLX-backed [`LunaEncoder`] — same role as `crate::encoder::LunaEncoder`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use crate::config::ModelConfig;
use super::graph::{build_forward_graph, ForwardSpec};
use super::prepare::{channel_wise_normalize, gather_channel_emb, prepare_tokens};
use super::rope_helpers::{build_rope_table, precompute_rope};
use super::weights::{
    apply_params, build_forward_params, build_prepare_params, load_safetensors, ParamMap,
};

/// Per-epoch output from LUNA inference.
#[derive(Clone, Debug)]
pub struct EpochEmbedding {
    pub output: Vec<f32>,
    pub shape: Vec<usize>,
    pub chan_pos: Vec<f32>,
    pub n_channels: usize,
}

/// Options for [`LunaEncoder::run_epoch`].
#[derive(Clone, Copy, Debug)]
pub struct RunEpochOpts {
    /// Apply per-channel z-score normalisation along time (default `true`).
    pub normalize: bool,
}

impl Default for RunEpochOpts {
    fn default() -> Self {
        Self { normalize: true }
    }
}

/// RLX LUNA encoder with per-shape compiled-graph cache.
pub struct LunaEncoder {
    pub model_cfg: ModelConfig,
    pub device: rlx::Device,

    forward_params: ParamMap,
    prepare_params: ParamMap,
    rope_table: Vec<f32>,

    session: rlx::Session,
    forward_cache: HashMap<u64, rlx::CompiledGraph>,
}

impl LunaEncoder {
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(
            hf_val.get("model").cloned().unwrap_or(hf_val),
        )
        .context("parsing model config")?;

        let t = std::time::Instant::now();
        let mut raw = load_safetensors(
            weights_path
                .to_str()
                .context("weights path not valid UTF-8")?,
        )?;
        let mut raw_prepare = raw.clone();
        let forward_params = build_forward_params(&mut raw, &model_cfg)?;
        let prepare_params = build_prepare_params(&mut raw_prepare)?;

        let head_dim = model_cfg.head_dim();
        let rope_table = build_rope_table(head_dim, 1024, 10_000.0);
        let session = rlx::Session::new(device);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((
            Self {
                model_cfg,
                device,
                forward_params,
                prepare_params,
                rope_table,
                session,
                forward_cache: HashMap::new(),
            },
            ms,
        ))
    }

    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        if c.num_classes > 0 {
            format!(
                "LUNA classifier (RLX, dev={:?})  embed_dim={}  classes={}",
                self.device, c.embed_dim, c.num_classes,
            )
        } else {
            format!(
                "LUNA encoder (RLX, dev={:?})  embed_dim={}  queries={}  depth={}  patch={}",
                self.device, c.embed_dim, c.num_queries, c.depth, c.patch_size,
            )
        }
    }

    fn spec(&self, b: usize, c: usize, t: usize) -> ForwardSpec {
        let cfg = &self.model_cfg;
        let s = t / cfg.patch_size;
        let hidden = cfg.hidden_dim();
        let nh_ca = cfg.num_heads;
        let nh_rot = cfg.total_heads();
        ForwardSpec {
            b,
            c,
            s,
            bt: b * s,
            d: cfg.embed_dim,
            q: cfg.num_queries,
            hidden,
            nh_ca,
            nh_rot,
            dh_ca: cfg.embed_dim / nh_ca,
            dh_rot: hidden / nh_rot,
            depth: cfg.depth,
            ff_ca: (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize,
            ff_rot: cfg.ffn_hidden_dim(),
            patch_size: cfg.patch_size,
            norm_eps: cfg.norm_eps as f32,
            num_classes: cfg.num_classes,
            nh_cls: cfg.num_heads,
        }
    }

    fn expand_queries(&self, bt: usize) -> Vec<f32> {
        let q = self.model_cfg.num_queries;
        let d = self.model_cfg.embed_dim;
        let embed = &self.forward_params["cross_attn.query_embed"];
        let flat = if embed.shape == vec![1, q, d] {
            embed.data.clone()
        } else {
            embed.data[..q * d].to_vec()
        };
        let mut out = vec![0f32; bt * q * d];
        for i in 0..bt {
            out[i * q * d..(i + 1) * q * d].copy_from_slice(&flat);
        }
        out
    }

    fn expand_agg_query(&self, b: usize) -> Vec<f32> {
        let hidden = self.model_cfg.hidden_dim();
        let embed = &self.forward_params["classifier.learned_agg"];
        let flat = if embed.shape == vec![1, 1, hidden] {
            embed.data.clone()
        } else {
            embed.data[..hidden].to_vec()
        };
        let mut out = vec![0f32; b * hidden];
        for i in 0..b {
            out[i * hidden..(i + 1) * hidden].copy_from_slice(&flat);
        }
        out
    }

    fn channel_emb_slice(
        &self,
        indices: Option<&[i32]>,
        b: usize,
        c: usize,
    ) -> Option<Vec<f32>> {
        let table = self.prepare_params.get("channel_emb.weight")?;
        let d = self.model_cfg.embed_dim;
        let idx = indices?;
        Some(gather_channel_emb(table, idx, b, c, d))
    }

    fn cache_key(&self, b: usize, c: usize, t: usize) -> u64 {
        (b as u64) << 40 | (c as u64) << 20 | (t as u64) | ((self.model_cfg.num_classes as u64) << 60)
    }

    fn compiled_for(&mut self, b: usize, c: usize, t: usize) -> &mut rlx::CompiledGraph {
        let key = self.cache_key(b, c, t);
        if !self.forward_cache.contains_key(&key) {
            let spec = self.spec(b, c, t);
            let graph = build_forward_graph(&spec);
            let mut compiled = self.session.compile(graph);
            apply_params(&mut compiled, &self.forward_params);
            self.forward_cache.insert(key, compiled);
        }
        self.forward_cache.get_mut(&key).expect("just inserted")
    }

    /// Run inference on one epoch.
    pub fn run_epoch(
        &mut self,
        signal: &[f32],
        chan_pos: &[f32],
        channel_indices: Option<&[i32]>,
        n_channels: usize,
        n_samples: usize,
    ) -> anyhow::Result<EpochEmbedding> {
        self.run_epoch_opts(
            signal,
            chan_pos,
            channel_indices,
            n_channels,
            n_samples,
            RunEpochOpts::default(),
        )
    }

    /// Run inference with explicit options (e.g. skip normalisation for parity vectors).
    pub fn run_epoch_opts(
        &mut self,
        signal: &[f32],
        chan_pos: &[f32],
        channel_indices: Option<&[i32]>,
        n_channels: usize,
        n_samples: usize,
        opts: RunEpochOpts,
    ) -> anyhow::Result<EpochEmbedding> {
        let b = 1usize;
        let c = n_channels;
        let t = n_samples;
        let patch_size = self.model_cfg.patch_size;
        let embed_dim = self.model_cfg.embed_dim;

        let mut sig = signal.to_vec();
        if opts.normalize {
            channel_wise_normalize(&mut sig, c, t);
        }

        let ch_emb = self.channel_emb_slice(channel_indices, b, c);
        let (x_tok, dec_q) = prepare_tokens(
            &sig,
            chan_pos,
            ch_emb.as_deref(),
            b,
            c,
            t,
            patch_size,
            embed_dim,
            &self.prepare_params,
        );
        self.run_forward_prepared(&x_tok, &dec_q, b, c, t, chan_pos)
    }

    /// Run the RLX graph from prepared `x_tokenized` / `decoder_queries` (skips CPU prepare).
    pub fn run_forward_prepared(
        &mut self,
        x_tokenized: &[f32],
        decoder_queries: &[f32],
        b: usize,
        c: usize,
        t: usize,
        chan_pos: &[f32],
    ) -> anyhow::Result<EpochEmbedding> {
        let num_classes = self.model_cfg.num_classes;
        let spec = self.spec(b, c, t);
        let queries = self.expand_queries(spec.bt);
        let head_dim = spec.dh_rot;
        let (cos, sin) = precompute_rope(&self.rope_table, head_dim, spec.s);
        let agg_query = if num_classes > 0 {
            Some(self.expand_agg_query(b))
        } else {
            None
        };

        let compiled = self.compiled_for(b, c, t);
        let mut inputs = vec![
            ("x_tokenized", x_tokenized),
            ("queries", queries.as_slice()),
            ("freqs_cos", cos.as_slice()),
            ("freqs_sin", sin.as_slice()),
        ];
        if let Some(ref agg) = agg_query {
            inputs.push(("agg_query", agg.as_slice()));
        } else {
            inputs.push(("decoder_queries", decoder_queries));
        }

        let outs = compiled.run(&inputs);
        let output = outs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("forward graph produced no output"))?;

        let shape = if num_classes > 0 {
            vec![num_classes]
        } else {
            vec![c, t]
        };

        Ok(EpochEmbedding {
            output,
            shape,
            chan_pos: chan_pos.to_vec(),
            n_channels: c,
        })
    }

    /// Convenience wrapper around [`super::io::RlxEpoch`].
    pub fn run_rlx_epoch(&mut self, ep: &super::io::RlxEpoch) -> anyhow::Result<EpochEmbedding> {
        self.run_epoch(
            &ep.signal,
            &ep.chan_pos,
            ep.channel_indices.as_deref(),
            ep.n_channels,
            ep.n_samples,
        )
    }
}
