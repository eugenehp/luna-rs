//! Standalone LUNA encoder — produce latent EEG embeddings.
//!
//! LUNA is an encoder-only model (no diffusion decoder like ZUNA).
//! The encoder produces:
//! - For pretraining: reconstructed signal [B, C, T]
//! - For classification: logits [B, num_classes]
//! - For embeddings: latent representations [B, S, Q*D]

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::prelude::*;

use crate::{
    config::{DataConfig, ModelConfig},
    data::{InputBatch, FifInfo, channel_wise_normalize},
    model::luna::Luna,
    model::rope::RotaryEmbedding,
    weights::load_model,
};

/// Per-epoch embedding produced by LUNA.
pub struct EpochEmbedding {
    /// Output values: row-major f32.
    /// - Reconstruction mode: [C, T]
    /// - Classification mode: \[num_classes\]
    pub output: Vec<f32>,
    /// Shape of the output.
    pub shape: Vec<usize>,
    /// Channel positions in metres: [C, 3].
    pub chan_pos: Vec<f32>,
    pub n_channels: usize,
}

/// Collection of per-epoch outputs.
pub struct EncodingResult {
    pub epochs: Vec<EpochEmbedding>,
    pub fif_info: Option<FifInfo>,
    pub ms_preproc: f64,
    pub ms_encode: f64,
}

impl EncodingResult {
    /// Load from a previously saved safetensors file.
    pub fn load_safetensors(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;

        let n_samples = {
            let v = st.tensor("n_samples")?;
            f32::from_le_bytes(v.data()[..4].try_into().unwrap()) as usize
        };

        let mut epochs = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let out_view = st.tensor(&format!("output_{i}"))?;
            let shape: Vec<usize> = out_view.shape().to_vec();
            let output: Vec<f32> = out_view.data().chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            let pos_view = st.tensor(&format!("chan_pos_{i}"))?;
            let n_channels = pos_view.shape()[0];
            let chan_pos: Vec<f32> = pos_view.data().chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            epochs.push(EpochEmbedding { output, shape, chan_pos, n_channels });
        }

        Ok(Self { epochs, fif_info: None, ms_preproc: 0.0, ms_encode: 0.0 })
    }

    /// Save to safetensors file.
    pub fn save_safetensors(&self, path: &str) -> anyhow::Result<()> {
        use safetensors::{Dtype, View};
        use std::borrow::Cow;

        struct RawTensor { data: Vec<u8>, shape: Vec<usize>, dtype: Dtype }
        impl View for RawTensor {
            fn dtype(&self)    -> Dtype         { self.dtype }
            fn shape(&self)    -> &[usize]      { &self.shape }
            fn data(&self)     -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize          { self.data.len() }
        }

        let f32_bytes = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

        let mut keys: Vec<String> = Vec::new();
        let mut tensors: Vec<RawTensor> = Vec::new();

        for (i, ep) in self.epochs.iter().enumerate() {
            keys.push(format!("output_{i}"));
            tensors.push(RawTensor {
                data: f32_bytes(&ep.output),
                shape: ep.shape.clone(),
                dtype: Dtype::F32,
            });

            keys.push(format!("chan_pos_{i}"));
            tensors.push(RawTensor {
                data: f32_bytes(&ep.chan_pos),
                shape: vec![ep.n_channels, 3],
                dtype: Dtype::F32,
            });
        }

        let n = self.epochs.len() as f32;
        keys.push("n_samples".into());
        tensors.push(RawTensor {
            data: f32_bytes(&[n]),
            shape: vec![1],
            dtype: Dtype::F32,
        });

        let pairs: Vec<(&str, RawTensor)> = keys.iter()
            .map(|s| s.as_str())
            .zip(tensors)
            .collect();
        let bytes = safetensors::serialize(pairs, None)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ── LunaEncoder ───────────────────────────────────────────────────────────────

/// LUNA encoder for EEG signal processing.
pub struct LunaEncoder<B: Backend> {
    model:     Luna<B>,
    rope:      RotaryEmbedding<B>,
    pub model_cfg: ModelConfig,
    pub data_cfg:  DataConfig,
    device:    B::Device,
}

impl<B: Backend> LunaEncoder<B> {
    /// Load model from config.json and weights safetensors.
    pub fn load(
        config_path:  &Path,
        weights_path: &Path,
        device:       B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(
            hf_val.get("model").cloned().unwrap_or(hf_val.clone())
        ).context("parsing model config")?;

        // RoPE for the temporal encoder
        let max_seqlen = 1024;  // generous upper bound
        let head_dim = model_cfg.hidden_dim() / model_cfg.total_heads();
        let rope = RotaryEmbedding::new(head_dim, max_seqlen, 10_000.0, &device);

        let t = Instant::now();
        let n_channel_names = 90;  // global vocabulary size (matches Python)
        let model = load_model::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            n_channel_names,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self { model, rope, model_cfg, data_cfg: DataConfig::default(), device }, ms))
    }

    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "LUNA  embed_dim={}  queries={}  depth={}  heads={}  patch={}  classes={}",
            c.embed_dim, c.num_queries, c.depth, c.num_heads, c.patch_size, c.num_classes,
        )
    }

    /// Run inference on a prepared InputBatch.
    pub fn run_batch(&self, batch: &InputBatch<B>) -> anyhow::Result<EpochEmbedding> {
        use crate::model::luna::LunaOutput;

        // Channel-wise z-score normalisation
        let signal = channel_wise_normalize(batch.signal.clone());

        let luna_output = self.model.forward(
            signal,
            batch.channel_locations.clone(),
            None,  // no mask at inference
            batch.channel_names.clone(),
            &self.rope,
        );

        // Extract the primary output tensor
        let output = match luna_output {
            LunaOutput::Classification { logits, .. } => logits,
            LunaOutput::Reconstruction { x_reconstructed, .. } => x_reconstructed,
        };

        let shape = output.dims().to_vec();
        let output_vec = output.squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("output→vec: {e:?}"))?;

        let chan_pos = batch.channel_locations.clone()
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("chan_pos→vec: {e:?}"))?;

        Ok(EpochEmbedding {
            output: output_vec,
            shape: shape[1..].to_vec(),  // remove batch dim
            chan_pos,
            n_channels: batch.n_channels,
        })
    }

    /// Run on multiple batches.
    pub fn run_batches(&self, batches: &[InputBatch<B>]) -> anyhow::Result<Vec<EpochEmbedding>> {
        batches.iter().map(|b| self.run_batch(b)).collect()
    }

    pub fn device(&self) -> &B::Device { &self.device }
}
