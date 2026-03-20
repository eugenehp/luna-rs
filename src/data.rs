/// Data preparation for LUNA inference (burn 0.20.1)
///
/// LUNA input: (B, C, T) signal + (B, C, 3) channel locations + optional channel name indices.

use burn::prelude::*;

/// A single prepared input for the LUNA model.
pub struct InputBatch<B: Backend> {
    /// EEG signal: [1, C, T] — z-scored and normalised.
    pub signal: Tensor<B, 3>,
    /// Channel 3D positions in metres: [1, C, 3].
    pub channel_locations: Tensor<B, 3>,
    /// Channel name indices into the global vocabulary: [1, C].
    pub channel_names: Option<Tensor<B, 2, Int>>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time samples.
    pub n_samples: usize,
}

/// Metadata from a FIF file.
pub struct FifInfo {
    pub ch_names: Vec<String>,
    pub ch_pos_mm: Vec<[f32; 3]>,
    pub sfreq: f32,
    pub n_times_raw: usize,
    pub duration_s: f32,
    pub n_epochs: usize,
    pub target_sfreq: f32,
    pub epoch_dur_s: f32,
}

/// Channel-wise z-score normalisation (matching Python `ChannelWiseNormalize`).
pub fn channel_wise_normalize<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);  // [B, C, 1]
    let diff = x.clone() - mean.clone();
    let var = (diff.clone() * diff.clone()).mean_dim(2);
    let std = (var + 1e-8).sqrt();
    (x - mean) / std
}

/// Build InputBatch from raw arrays with explicit channel name indices.
pub fn build_batch<B: Backend>(
    signal: Vec<f32>,       // [C, T] row-major
    positions: Vec<f32>,    // [C, 3] row-major
    channel_indices: Option<Vec<i64>>,  // [C] indices into CHANNEL_VOCAB
    n_channels: usize,
    n_samples: usize,
    device: &B::Device,
) -> InputBatch<B> {
    let signal = Tensor::<B, 2>::from_data(
        TensorData::new(signal, vec![n_channels, n_samples]), device,
    ).unsqueeze_dim::<3>(0);  // [1, C, T]

    let channel_locations = Tensor::<B, 2>::from_data(
        TensorData::new(positions, vec![n_channels, 3]), device,
    ).unsqueeze_dim::<3>(0);  // [1, C, 3]

    let channel_names = channel_indices.map(|idx| {
        Tensor::<B, 1, Int>::from_data(
            TensorData::new(idx, vec![n_channels]), device,
        ).unsqueeze_dim::<2>(0)  // [1, C]
    });

    InputBatch {
        signal,
        channel_locations,
        channel_names,
        n_channels,
        n_samples,
    }
}

/// Build InputBatch from channel name strings.
///
/// Automatically looks up:
/// - Channel vocabulary indices from `CHANNEL_VOCAB`
/// - 3D electrode positions (bipolar midpoints for names like "FP1-F7")
///
/// This is the recommended way to build batches for LUNA inference.
pub fn build_batch_named<B: Backend>(
    signal: Vec<f32>,         // [C, T] row-major
    channel_names: &[&str],   // e.g. ["FP1-F7", "F7-T3", ...]
    n_samples: usize,
    device: &B::Device,
) -> InputBatch<B> {
    let n_channels = channel_names.len();

    // Look up channel vocabulary indices
    let indices = crate::channel_vocab::channel_indices_unwrap(channel_names);

    // Look up 3D positions (bipolar midpoint or unipolar)
    let positions: Vec<f32> = channel_names.iter()
        .flat_map(|name| {
            crate::channel_positions::bipolar_channel_xyz(name)
                .unwrap_or([0.0, 0.0, 0.0])
                .to_vec()
        })
        .collect();

    build_batch(signal, positions, Some(indices), n_channels, n_samples, device)
}
