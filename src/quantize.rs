//! INT8 post-training quantization for LUNA.
//!
//! Provides dynamic per-tensor symmetric INT8 quantization for inference.
//! Weights are quantized offline; activations are quantized dynamically at runtime.
//!
//! # Approach
//!
//! This implements **weight-only quantization**: model weights are stored as INT8
//! with per-tensor f32 scale factors. Activations remain in f32/f64. Matrix
//! multiplications are: dequantize weight → f32 matmul. This reduces model size
//! by ~4× and improves cache utilization, at the cost of a small accuracy drop.
//!
//! Burn 0.20.1 has native quantization support (`Tensor::quantize_dynamic`),
//! but it operates on the backend level and requires specific backend support.
//! We implement a simpler approach that works on any backend: quantize/dequantize
//! as a data transformation, keeping all compute in f32.
//!
//! # Usage
//!
//! ```rust,ignore
//! use luna_rs::quantize::{quantize_tensor, dequantize_tensor, QuantizedWeight};
//!
//! // Quantize a weight tensor
//! let weight_f32: Vec<f32> = vec![0.1, -0.5, 0.3, 0.8];
//! let qw = QuantizedWeight::from_f32(&weight_f32);
//! println!("Compressed: {} bytes → {} bytes", weight_f32.len() * 4, qw.data.len());
//!
//! // Dequantize back
//! let recovered = qw.to_f32();
//! ```

use anyhow::Result;
use std::collections::HashMap;

/// A single quantized weight tensor.
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// INT8 quantized values.
    pub data: Vec<i8>,
    /// Per-tensor scale factor: `float_value = int8_value * scale`.
    pub scale: f32,
    /// Original tensor shape.
    pub shape: Vec<usize>,
}

impl QuantizedWeight {
    /// Quantize a f32 tensor to symmetric INT8.
    ///
    /// Uses symmetric quantization: `scale = max(|tensor|) / 127`.
    /// Zero-point is always 0 (symmetric around zero).
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Self {
        let max_abs = data.iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        let scale = if max_abs < 1e-10 { 1e-10 } else { max_abs / 127.0 };
        let inv_scale = 1.0 / scale;

        let quantized: Vec<i8> = data.iter()
            .map(|&v| {
                let q = (v * inv_scale).round();
                q.clamp(-128.0, 127.0) as i8
            })
            .collect();

        Self { data: quantized, scale, shape }
    }

    /// Dequantize back to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter()
            .map(|&v| v as f32 * self.scale)
            .collect()
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Compressed size in bytes (INT8 data + scale + shape).
    pub fn size_bytes(&self) -> usize {
        self.data.len() + 4 + self.shape.len() * 8 // i8 data + f32 scale + usize shape
    }
}

/// Collection of quantized model weights.
pub struct QuantizedModel {
    /// Quantized weight tensors, keyed by parameter name.
    pub weights: HashMap<String, QuantizedWeight>,
    /// Parameters that were NOT quantized (e.g., biases, norms, embeddings).
    pub unquantized: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl QuantizedModel {
    /// Quantize a model's weights from a safetensors file.
    ///
    /// Strategy:
    /// - 2D weight matrices (Linear weights, Conv weights) → INT8
    /// - 1D tensors (biases, LayerNorm gamma/beta, embeddings) → keep f32
    /// - Embedding tables → keep f32 (lookup tables don't benefit from quantization)
    /// - Small tensors (< 32 elements) → keep f32
    pub fn from_safetensors(path: &str) -> Result<Self> {
        let wm = crate::weights::WeightMap::from_file(path)?;
        let mut weights = HashMap::new();
        let mut unquantized = HashMap::new();

        let mut total_f32_bytes = 0usize;
        let mut total_q8_bytes = 0usize;
        let mut n_quantized = 0usize;
        let mut n_kept = 0usize;

        for (key, (data, shape)) in &wm.tensors {
            let numel = data.len();
            total_f32_bytes += numel * 4;

            // Quantize: 2D matrices with >= 32 elements that aren't embeddings
            let should_quantize = shape.len() >= 2
                && numel >= 32
                && !key.contains("embedding")
                && !key.contains("channel_emb");

            if should_quantize {
                let qw = QuantizedWeight::from_f32(data, shape.clone());
                total_q8_bytes += qw.size_bytes();
                weights.insert(key.clone(), qw);
                n_quantized += 1;
            } else {
                total_q8_bytes += numel * 4; // kept as f32
                unquantized.insert(key.clone(), (data.clone(), shape.clone()));
                n_kept += 1;
            }
        }

        println!("Quantization summary:");
        println!("  Quantized:   {n_quantized} tensors");
        println!("  Kept f32:    {n_kept} tensors");
        println!("  Size: {:.2} MB → {:.2} MB ({:.1}× compression)",
            total_f32_bytes as f64 / 1e6,
            total_q8_bytes as f64 / 1e6,
            total_f32_bytes as f64 / total_q8_bytes as f64,
        );

        Ok(Self { weights, unquantized })
    }

    /// Save quantized model to a custom binary format.
    ///
    /// Format: JSON header + binary data.
    pub fn save(&self, path: &str) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Header: key → (offset, length, scale, shape, is_quantized)
        #[derive(serde::Serialize)]
        struct Entry {
            offset: usize,
            length: usize,
            scale: Option<f32>,
            shape: Vec<usize>,
            quantized: bool,
        }

        let mut entries: HashMap<String, Entry> = HashMap::new();
        let mut data_buf: Vec<u8> = Vec::new();

        // Write quantized weights
        for (key, qw) in &self.weights {
            let offset = data_buf.len();
            data_buf.extend_from_slice(bytemuck::cast_slice(&qw.data));
            entries.insert(key.clone(), Entry {
                offset,
                length: qw.data.len(),
                scale: Some(qw.scale),
                shape: qw.shape.clone(),
                quantized: true,
            });
        }

        // Write unquantized weights
        for (key, (data, shape)) in &self.unquantized {
            let offset = data_buf.len();
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            data_buf.extend_from_slice(&bytes);
            entries.insert(key.clone(), Entry {
                offset,
                length: data.len(),
                scale: None,
                shape: shape.clone(),
                quantized: false,
            });
        }

        // Write header as JSON + newline separator
        let header = serde_json::to_string(&entries)?;
        file.write_all(header.as_bytes())?;
        file.write_all(b"\n")?;
        file.write_all(&data_buf)?;

        Ok(())
    }

    /// Load a quantized model and dequantize all weights back to f32.
    ///
    /// Returns a WeightMap compatible with `load_model`.
    pub fn load_and_dequantize(path: &str) -> Result<crate::weights::WeightMap> {
        let content = std::fs::read(path)?;

        // Find the JSON header (ends at first newline)
        let newline_pos = content.iter().position(|&b| b == b'\n')
            .ok_or_else(|| anyhow::anyhow!("invalid quantized model format"))?;

        let header_str = std::str::from_utf8(&content[..newline_pos])?;
        let data = &content[newline_pos + 1..];

        #[derive(serde::Deserialize)]
        struct Entry {
            offset: usize,
            length: usize,
            scale: Option<f32>,
            shape: Vec<usize>,
            quantized: bool,
        }

        let entries: HashMap<String, Entry> = serde_json::from_str(header_str)?;
        let mut tensors = HashMap::new();

        for (key, entry) in &entries {
            let f32_data = if entry.quantized {
                // Dequantize: i8 * scale → f32
                let scale = entry.scale.unwrap_or(1.0);
                let i8_data = &data[entry.offset..entry.offset + entry.length];
                i8_data.iter()
                    .map(|&b| b as i8 as f32 * scale)
                    .collect()
            } else {
                // Read f32 directly
                let byte_offset = entry.offset;
                let byte_len = entry.length * 4;
                data[byte_offset..byte_offset + byte_len]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            };

            tensors.insert(key.clone(), (f32_data, entry.shape.clone()));
        }

        Ok(crate::weights::WeightMap { tensors })
    }

    /// Compute quantization error statistics.
    pub fn error_stats(path: &str) -> Result<()> {
        let wm = crate::weights::WeightMap::from_file(path)?;
        let mut total_mse = 0.0f64;
        let mut total_elements = 0usize;
        let mut max_err = 0.0f32;

        for (key, (data, shape)) in &wm.tensors {
            if shape.len() < 2 || data.len() < 32 { continue; }
            if key.contains("embedding") || key.contains("channel_emb") { continue; }

            let qw = QuantizedWeight::from_f32(data, shape.clone());
            let recovered = qw.to_f32();

            for (orig, rec) in data.iter().zip(recovered.iter()) {
                let err = (orig - rec).abs();
                max_err = max_err.max(err);
                total_mse += (err as f64).powi(2);
                total_elements += 1;
            }
        }

        let rmse = (total_mse / total_elements as f64).sqrt();
        println!("INT8 quantization error:");
        println!("  RMSE:      {rmse:.6}");
        println!("  Max error: {max_err:.6}");
        println!("  Elements:  {total_elements}");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dequantize_round_trip() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75];
        let qw = QuantizedWeight::from_f32(&data, vec![7]);

        let recovered = qw.to_f32();
        assert_eq!(recovered.len(), data.len());

        // Check round-trip error is small
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            assert!(err < 0.01, "error {err} too large for value {orig}");
        }
    }

    #[test]
    fn quantize_zero_tensor() {
        let data = vec![0.0f32; 100];
        let qw = QuantizedWeight::from_f32(&data, vec![10, 10]);
        let recovered = qw.to_f32();
        for v in &recovered {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn compression_ratio() {
        let data = vec![0.1f32; 1024];
        let qw = QuantizedWeight::from_f32(&data, vec![32, 32]);
        // f32: 4096 bytes, INT8: ~1024 bytes + overhead
        assert!(qw.size_bytes() < 4096 / 2, "should be at least 2× smaller");
    }

    #[test]
    fn symmetric_range() {
        let data = vec![-1.0f32, 1.0, 0.0, 0.5, -0.5];
        let qw = QuantizedWeight::from_f32(&data, vec![5]);
        // scale should be 1.0/127
        assert!((qw.scale - 1.0 / 127.0).abs() < 1e-6);
        // -1.0 should map to -127, 1.0 to 127
        assert_eq!(qw.data[0], -127);
        assert_eq!(qw.data[1], 127);
        assert_eq!(qw.data[2], 0);
    }
}
