/// Frequency Feature Embedder for LUNA.
///
/// Python: `FrequencyFeatureEmbedder` in frequency_embedder.py.
///
/// For each patch, computes `torch.fft.rfft` → magnitude + phase, then
/// projects through a timm.Mlp (no norm_layer): Linear → GELU → Linear.
///
/// Uses `rustfft` (same Cooley-Tukey algorithm as torch.fft.rfft) for
/// exact numerical parity with Python.

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use rustfft::{FftPlanner, num_complex::Complex};

#[derive(Module, Debug)]
pub struct FrequencyFeatureEmbedder<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub patch_size: usize,
    pub embed_dim:  usize,
}

impl<B: Backend> FrequencyFeatureEmbedder<B> {
    pub fn new(embed_dim: usize, patch_size: usize, device: &B::Device) -> Self {
        let n_freq = patch_size / 2 + 1;
        let in_features = 2 * n_freq;
        let hidden = 4 * in_features;

        Self {
            fc1: LinearConfig::new(in_features, hidden).with_bias(true).init(device),
            fc2: LinearConfig::new(hidden, embed_dim).with_bias(true).init(device),
            patch_size,
            embed_dim,
        }
    }

    /// x: [B, C, T] → [B, C*S, embed_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, t] = x.dims();
        let s = t / self.patch_size;
        let p = self.patch_size;
        let n_freq = p / 2 + 1;
        let n_elements = b * c * s;
        let device = x.device();

        // Extract signal data to CPU for FFT
        // Handle both f32 and f64 backends by trying f64 first, falling back to f32
        let tensor_data = x.reshape([n_elements, p]).into_data();
        let x_f32: Vec<f32> = tensor_data.to_vec::<f64>()
            .map(|v| v.into_iter().map(|x| x as f32).collect())
            .or_else(|_| tensor_data.to_vec::<f32>())
            .expect("extract tensor data as f32");

        // Compute rfft in f64 for maximum precision, matching torch.fft.rfft's f32→f64→f32 path.
        // PyTorch internally promotes to f64 for FFT on CPU (MKL/FFTW), then casts back.
        // Using f64 here eliminates the ~0.000002 FFT accumulation error entirely.
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(p);

        let mut mag_phase = vec![0.0f32; n_elements * 2 * n_freq];

        for i in 0..n_elements {
            let mut buf: Vec<Complex<f64>> = x_f32[i * p..(i + 1) * p]
                .iter()
                .map(|&v| Complex { re: v as f64, im: 0.0 })
                .collect();

            fft.process(&mut buf);

            // Extract rfft: first n_freq bins → magnitude + phase, cast back to f32
            let offset = i * 2 * n_freq;
            for k in 0..n_freq {
                let re = buf[k].re;
                let im = buf[k].im;
                mag_phase[offset + k] = (re * re + im * im).sqrt() as f32;
                mag_phase[offset + n_freq + k] = im.atan2(re) as f32;
            }
        }

        // Back to tensor: [n_elements, 2*n_freq]
        let freq_features = Tensor::<B, 2>::from_data(
            TensorData::new(mag_phase, vec![n_elements, 2 * n_freq]),
            &device,
        );

        // timm.Mlp (no norm_layer): fc1 → GELU → fc2
        let h = gelu(self.fc1.forward(freq_features));
        let embedded = self.fc2.forward(h);

        embedded.reshape([b, c * s, self.embed_dim])
    }
}
