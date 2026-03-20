//! End-to-end EEG preprocessing for LUNA inference.
//!
//! Bridges `exg` / `exg-luna` preprocessing with `luna-rs` `InputBatch`:
//!
//! ```text
//! .edf / .fif / .csv
//!   │
//!   ├─ exg: read raw data
//!   ├─ exg-luna: bandpass → notch → resample → bipolar → epoch
//!   └─ luna-rs: channel vocab lookup → InputBatch
//! ```
//!
//! # Quick start
//!
//! ```rust,ignore
//! use luna_rs::preprocessing::load_edf;
//!
//! let (batches, info) = load_edf::<B>("recording.edf", &device)?;
//! for batch in &batches {
//!     let result = encoder.run_batch(batch)?;
//! }
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use burn::prelude::*;

use crate::channel_positions::bipolar_channel_xyz;
use crate::channel_vocab;
use crate::data::InputBatch;

/// Preprocessing metadata returned alongside batches.
pub struct PreprocInfo {
    /// Channel names after bipolar montage (e.g. "FP1-F7", ...).
    pub ch_names: Vec<String>,
    /// Number of channels after montage.
    pub n_channels: usize,
    /// Number of epochs produced.
    pub n_epochs: usize,
    /// Source sampling rate (Hz).
    pub src_sfreq: f32,
    /// Target sampling rate (Hz) — always 256.
    pub target_sfreq: f32,
    /// Epoch duration (seconds) — always 5.0.
    pub epoch_dur: f32,
}

/// Build `InputBatch`es from exg-luna preprocessed epochs.
fn epochs_to_batches<B: Backend>(
    epochs: Vec<(ndarray::Array2<f32>, Vec<String>)>,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    if epochs.is_empty() {
        anyhow::bail!("preprocessing produced zero epochs");
    }

    let ch_names = epochs[0].1.clone();
    let n_channels = ch_names.len();
    let n_samples = epochs[0].0.ncols();

    // Look up channel positions and vocab indices
    let positions: Vec<f32> = ch_names.iter()
        .flat_map(|name| {
            bipolar_channel_xyz(name)
                .unwrap_or([0.0, 0.0, 0.0])
                .to_vec()
        })
        .collect();

    let vocab_indices: Option<Vec<i64>> = {
        let indices: Vec<Option<usize>> = ch_names.iter()
            .map(|n| channel_vocab::channel_index(n))
            .collect();
        if indices.iter().all(|i| i.is_some()) {
            Some(indices.iter().map(|i| i.unwrap() as i64).collect())
        } else {
            None
        }
    };

    let n_epochs = epochs.len();
    let mut batches = Vec::with_capacity(n_epochs);

    for (epoch_data, _names) in &epochs {
        let signal: Vec<f32> = epoch_data.iter().copied().collect();
        batches.push(crate::data::build_batch::<B>(
            signal,
            positions.clone(),
            vocab_indices.clone(),
            n_channels,
            n_samples,
            device,
        ));
    }

    let info = PreprocInfo {
        ch_names,
        n_channels,
        n_epochs,
        src_sfreq: 256.0, // after preprocessing
        target_sfreq: 256.0,
        epoch_dur: 5.0,
    };

    Ok((batches, info))
}

/// Load and preprocess an EDF file for LUNA inference.
///
/// Applies the full LUNA pipeline:
/// 1. Read EDF → raw signal + channel names
/// 2. Channel rename (strip "EEG ", "-REF", "-LE")
/// 3. Pick standard 10-20 channels
/// 4. Bandpass filter 0.1–75 Hz
/// 5. Notch filter 60 Hz
/// 6. Resample to 256 Hz
/// 7. TCP bipolar montage (22 channels)
/// 8. Epoch into 5s windows (1280 samples)
///
/// Returns `InputBatch`es ready for `LunaEncoder::run_batch()`.
pub fn load_edf<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    load_edf_with_config(path, &exg_luna::LunaPipelineConfig::default(), device)
}

/// Load and preprocess an EDF file with custom pipeline config.
pub fn load_edf_with_config<B: Backend>(
    path: &Path,
    cfg: &exg_luna::LunaPipelineConfig,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    // Read EDF
    let raw = exg::edf::open_raw_edf(path)
        .with_context(|| format!("opening EDF: {}", path.display()))?;
    let data = raw.read_all_data()
        .with_context(|| format!("reading EDF data: {}", path.display()))?;
    let ch_names: Vec<String> = raw.channel_names();
    let sfreq = raw.header.sample_rate;

    // Run LUNA preprocessing pipeline
    let epochs = exg_luna::preprocess_luna(data, &ch_names, sfreq, cfg)
        .with_context(|| "LUNA preprocessing failed")?;

    let mut info_result = epochs_to_batches(epochs, device)?;
    info_result.1.src_sfreq = sfreq;
    Ok(info_result)
}

/// Load and preprocess a FIF file for LUNA inference.
///
/// Same pipeline as [`load_edf`] but reads FIF format.
pub fn load_fif<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    load_fif_with_config(path, &exg_luna::LunaPipelineConfig::default(), device)
}

/// Load and preprocess a FIF file with custom pipeline config.
pub fn load_fif_with_config<B: Backend>(
    path: &Path,
    cfg: &exg_luna::LunaPipelineConfig,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    let raw = exg::fiff::raw::open_raw(path)
        .with_context(|| format!("opening FIF: {}", path.display()))?;
    let data = raw.read_all_data()
        .with_context(|| format!("reading FIF data: {}", path.display()))?;
    let ch_names: Vec<String> = raw.info.chs.iter().map(|ch| ch.name.clone()).collect();
    let sfreq = raw.info.sfreq as f32;
    let data_f32 = data.mapv(|v| v as f32);

    let epochs = exg_luna::preprocess_luna(data_f32, &ch_names, sfreq, cfg)
        .with_context(|| "LUNA preprocessing failed")?;

    let mut info_result = epochs_to_batches(epochs, device)?;
    info_result.1.src_sfreq = sfreq;
    Ok(info_result)
}

/// Load preprocessed LUNA epochs from exg-luna safetensors format.
///
/// This reads files exported by `exg_luna::export_luna_epochs`.
pub fn load_luna_epochs<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    let epochs_data = exg_luna::load_luna_epochs(path)
        .with_context(|| format!("loading LUNA epochs: {}", path.display()))?;

    // Convert exg_luna::LunaEpoch → (Array2, Vec<String>)
    let epochs: Vec<(ndarray::Array2<f32>, Vec<String>)> = epochs_data.into_iter()
        .map(|e| (e.signal, e.channel_names))
        .collect();

    epochs_to_batches(epochs, device)
}

/// Load EEG from a CSV file, preprocess, and produce InputBatches.
///
/// The CSV must have channel names in the header that match the
/// standard 10-20 electrode names (possibly with "EEG " prefix and "-REF" suffix).
/// Applies the full LUNA pipeline (bandpass, notch, resample, bipolar montage, epoch).
pub fn load_csv_and_preprocess<B: Backend>(
    path: &Path,
    sample_rate: f32,
    device: &B::Device,
) -> Result<(Vec<InputBatch<B>>, PreprocInfo)> {
    let (data, ch_names, detected_sfreq) = exg::csv::read_eeg(path)
        .with_context(|| format!("reading CSV: {}", path.display()))?;

    let sfreq = if detected_sfreq > 0.0 { detected_sfreq } else { sample_rate };
    let ch_strings: Vec<String> = ch_names;

    let cfg = exg_luna::LunaPipelineConfig::default();
    let epochs = exg_luna::preprocess_luna(data, &ch_strings, sfreq, &cfg)
        .with_context(|| "LUNA preprocessing of CSV failed")?;

    let mut info_result = epochs_to_batches(epochs, device)?;
    info_result.1.src_sfreq = sfreq;
    Ok(info_result)
}
