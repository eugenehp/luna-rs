//! CSV loader for EEG data.
//!
//! Reads CSV files with a timestamp column and EEG channel columns,
//! looks up electrode positions, and produces `InputBatch` structs
//! ready for LUNA inference.
//!
//! ## CSV format
//!
//! ```text
//! timestamp,FP1-F7,F7-T3,T3-T5,...
//! 0.000000,2.07e-05,8.38e-07,...
//! 0.003906,...
//! ```
//!
//! - First column: timestamps in seconds (name is auto-detected)
//! - Remaining columns: EEG channel values (volts)
//! - Lines starting with `#` are ignored
//!
//! ## Epoching
//!
//! The CSV is split into non-overlapping 5-second epochs (1280 samples at
//! 256 Hz). Trailing samples that don't fill a complete epoch are discarded.
//!
//! ## Feature gate
//!
//! This module is always compiled (no external dependencies beyond std).

use std::path::Path;
use anyhow::{bail, Context};
use burn::prelude::*;

use crate::channel_positions::bipolar_channel_xyz;
use crate::channel_vocab;
use crate::data::InputBatch;

/// Metadata returned alongside batches.
pub struct CsvInfo {
    pub ch_names: Vec<String>,
    pub sample_rate: f32,
    pub n_samples_raw: usize,
    pub duration_s: f32,
    pub n_epochs: usize,
}

/// Load EEG from a CSV file and split into 5-second epochs.
///
/// Each epoch becomes an `InputBatch` with:
/// - Signal: channel-wise data (NOT normalized — call `channel_wise_normalize` before inference)
/// - Channel locations: 3D positions from the montage database
/// - Channel names: vocabulary indices for the reconstruction head
///
/// `sample_rate` must match the CSV data (typically 256.0 Hz).
/// `epoch_samples` is the number of samples per epoch (typically 1280 = 5s × 256Hz).
pub fn load_from_csv<B: Backend>(
    path: &Path,
    sample_rate: f32,
    epoch_samples: usize,
    device: &B::Device,
) -> anyhow::Result<(Vec<InputBatch<B>>, CsvInfo)> {
    let (ch_names, data) = parse_csv(path)?;
    let n_ch = ch_names.len();
    let n_t = data[0].len();
    let duration_s = n_t as f32 / sample_rate;

    if n_t < epoch_samples {
        bail!("CSV has {n_t} samples ({duration_s:.2}s), need at least {epoch_samples} for one epoch");
    }

    // Look up positions and vocab indices
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
            None  // some channels not in vocab — skip channel embeddings
        }
    };

    // Epoch
    let n_epochs = n_t / epoch_samples;
    let mut batches = Vec::with_capacity(n_epochs);

    for e in 0..n_epochs {
        let start = e * epoch_samples;
        let end = start + epoch_samples;

        // Extract epoch signal [C, epoch_samples] row-major
        let mut signal = Vec::with_capacity(n_ch * epoch_samples);
        for ch in 0..n_ch {
            signal.extend_from_slice(&data[ch][start..end]);
        }

        batches.push(crate::data::build_batch::<B>(
            signal,
            positions.clone(),
            vocab_indices.clone(),
            n_ch,
            epoch_samples,
            device,
        ));
    }

    let info = CsvInfo {
        ch_names,
        sample_rate,
        n_samples_raw: n_t,
        duration_s,
        n_epochs,
    };

    Ok((batches, info))
}

/// Parse a CSV file into (channel_names, data[C][T]).
fn parse_csv(path: &Path) -> anyhow::Result<(Vec<String>, Vec<Vec<f32>>)> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;

    let mut lines = content.lines()
        .filter(|l| { let t = l.trim(); !t.is_empty() && !t.starts_with('#') });

    // Header
    let header_line = lines.next()
        .ok_or_else(|| anyhow::anyhow!("CSV is empty"))?;
    let header: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();
    anyhow::ensure!(header.len() >= 2, "need at least timestamp + one channel");

    // Detect timestamp column
    let ts_col = header.iter().position(|h| {
        let n = h.to_ascii_lowercase();
        n.contains("time") || n == "t" || n == "ts"
    }).unwrap_or(0);

    let ch_names: Vec<String> = header.iter().enumerate()
        .filter(|&(i, _)| i != ts_col)
        .map(|(_, h)| h.to_string())
        .collect();
    let n_ch = ch_names.len();

    // Data
    let mut columns: Vec<Vec<f32>> = vec![Vec::new(); n_ch];
    for (row_idx, line) in lines.enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        anyhow::ensure!(parts.len() == header.len(),
            "row {row_idx}: expected {} columns, got {}", header.len(), parts.len());

        let mut ch_idx = 0;
        for (i, s) in parts.iter().enumerate() {
            if i == ts_col { continue; }
            let v: f32 = s.trim().parse()
                .with_context(|| format!("row {row_idx} col {i}: cannot parse '{}'", s.trim()))?;
            columns[ch_idx].push(v);
            ch_idx += 1;
        }
    }

    anyhow::ensure!(!columns[0].is_empty(), "CSV has no data rows");
    Ok((ch_names, columns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_csv_basic() {
        let dir = std::env::temp_dir();
        let path = dir.join("luna_test.csv");
        std::fs::write(&path, "timestamp,FP1-F7,F7-T3\n0.0,1e-5,2e-5\n0.004,3e-5,4e-5\n").unwrap();

        let (names, data) = parse_csv(&path).unwrap();
        assert_eq!(names, ["FP1-F7", "F7-T3"]);
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].len(), 2);
        assert!((data[0][0] - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn parse_csv_skips_comments() {
        let dir = std::env::temp_dir();
        let path = dir.join("luna_test_comments.csv");
        std::fs::write(&path, "# comment\ntimestamp,C3-CZ\n0.0,0.5\n0.004,-0.3\n").unwrap();

        let (names, data) = parse_csv(&path).unwrap();
        assert_eq!(names, ["C3-CZ"]);
        assert_eq!(data[0].len(), 2);
    }
}
