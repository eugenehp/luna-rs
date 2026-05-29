//! RLX-side epoch I/O — preprocessing without Burn tensors.

use std::path::Path;

use anyhow::{Context, Result};
use exg_luna::LunaPipelineConfig;

use crate::channel_positions::bipolar_channel_xyz;
use crate::channel_vocab;

/// One preprocessed epoch ready for [`super::LunaEncoder::run_epoch`].
#[derive(Clone, Debug)]
pub struct RlxEpoch {
    /// EEG signal `[C, T]` row-major (one batch item).
    pub signal: Vec<f32>,
    /// Channel positions `[C, 3]` row-major.
    pub chan_pos: Vec<f32>,
    /// Vocabulary indices `[C]` when every channel name is known.
    pub channel_indices: Option<Vec<i32>>,
    pub n_channels: usize,
    pub n_samples: usize,
}

/// Metadata from a preprocessing run.
#[derive(Clone, Debug)]
pub struct PreprocInfo {
    pub ch_names: Vec<String>,
    pub n_channels: usize,
    pub n_epochs: usize,
    pub target_sfreq: f32,
    pub epoch_dur: f32,
}

fn epochs_to_rlx(
    epochs: Vec<(ndarray::Array2<f32>, Vec<String>)>,
) -> Result<(Vec<RlxEpoch>, PreprocInfo)> {
    if epochs.is_empty() {
        anyhow::bail!("preprocessing produced zero epochs");
    }

    let ch_names = epochs[0].1.clone();
    let n_channels = ch_names.len();
    let n_samples = epochs[0].0.ncols();

    let positions: Vec<f32> = ch_names
        .iter()
        .flat_map(|name| bipolar_channel_xyz(name).unwrap_or([0.0, 0.0, 0.0]).to_vec())
        .collect();

    let channel_indices: Option<Vec<i32>> = {
        let indices: Vec<Option<usize>> = ch_names
            .iter()
            .map(|n| channel_vocab::channel_index(n))
            .collect();
        if indices.iter().all(|i| i.is_some()) {
            Some(
                indices
                    .iter()
                    .map(|i| i.unwrap() as i32)
                    .collect(),
            )
        } else {
            None
        }
    };

    let n_epochs = epochs.len();
    let mut out = Vec::with_capacity(n_epochs);
    for (epoch_data, _) in &epochs {
        out.push(RlxEpoch {
            signal: epoch_data.iter().copied().collect(),
            chan_pos: positions.clone(),
            channel_indices: channel_indices.clone(),
            n_channels,
            n_samples,
        });
    }

    Ok((
        out,
        PreprocInfo {
            ch_names,
            n_channels,
            n_epochs,
            target_sfreq: 256.0,
            epoch_dur: 5.0,
        },
    ))
}

/// Load and preprocess an EDF file into RLX epochs.
pub fn load_edf(path: &Path) -> Result<(Vec<RlxEpoch>, PreprocInfo)> {
    load_edf_with_config(path, &LunaPipelineConfig::default())
}

/// Load and preprocess an EDF file with a custom pipeline config.
pub fn load_edf_with_config(
    path: &Path,
    cfg: &LunaPipelineConfig,
) -> Result<(Vec<RlxEpoch>, PreprocInfo)> {
    let raw = exg::edf::open_raw_edf(path)
        .with_context(|| format!("opening EDF: {}", path.display()))?;
    let data = raw
        .read_all_data()
        .with_context(|| format!("reading EDF data: {}", path.display()))?;
    let ch_names: Vec<String> = raw.channel_names();
    let sfreq = raw.header.sample_rate;

    let epochs = exg_luna::preprocess_luna(data, &ch_names, sfreq, cfg)
        .with_context(|| "LUNA preprocessing failed")?;
    epochs_to_rlx(epochs)
}

/// Load and preprocess a FIF file into RLX epochs.
pub fn load_fif(path: &Path) -> Result<(Vec<RlxEpoch>, PreprocInfo)> {
    load_fif_with_config(path, &LunaPipelineConfig::default())
}

/// Load and preprocess a FIF file with a custom pipeline config.
pub fn load_fif_with_config(
    path: &Path,
    cfg: &LunaPipelineConfig,
) -> Result<(Vec<RlxEpoch>, PreprocInfo)> {
    let raw = exg::fiff::raw::open_raw(path)
        .with_context(|| format!("opening FIF: {}", path.display()))?;
    let data = raw
        .read_all_data()
        .with_context(|| format!("reading FIF data: {}", path.display()))?;
    let ch_names: Vec<String> = raw.info.chs.iter().map(|ch| ch.name.clone()).collect();
    let sfreq = raw.info.sfreq as f32;
    let data_f32 = data.mapv(|v| v as f32);

    let epochs = exg_luna::preprocess_luna(data_f32, &ch_names, sfreq, cfg)
        .with_context(|| "LUNA preprocessing failed")?;
    epochs_to_rlx(epochs)
}

/// Write RLX epoch outputs to a safetensors file.
pub fn save_epochs(
    epochs: &[super::EpochEmbedding],
    path: &str,
) -> Result<()> {
    use safetensors::{Dtype, View};
    use std::borrow::Cow;

    struct RawTensor {
        data: Vec<u8>,
        shape: Vec<usize>,
    }
    impl View for RawTensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

    let mut keys = Vec::new();
    let mut tensors = Vec::new();
    for (i, ep) in epochs.iter().enumerate() {
        keys.push(format!("output_{i}"));
        tensors.push(RawTensor {
            data: f32_bytes(&ep.output),
            shape: ep.shape.clone(),
        });
        keys.push(format!("chan_pos_{i}"));
        tensors.push(RawTensor {
            data: f32_bytes(&ep.chan_pos),
            shape: vec![ep.n_channels, 3],
        });
    }
    keys.push("n_samples".into());
    tensors.push(RawTensor {
        data: f32_bytes(&[epochs.len() as f32]),
        shape: vec![1],
    });

    let pairs: Vec<(&str, RawTensor)> = keys.iter().map(|s| s.as_str()).zip(tensors).collect();
    std::fs::write(path, safetensors::serialize(pairs, None)?)?;
    Ok(())
}
