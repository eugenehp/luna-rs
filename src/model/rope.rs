/// Rotary Positional Embeddings for LUNA's temporal encoder.
///
/// LUNA uses standard 1-D RoPE (not 4-D axial like ZUNA).
/// Each position in the sequence gets a rotation applied to Q and K.
///
/// Python: `rotary_embedding_torch.RotaryEmbedding(dim=head_dim, learned_freq=False)`

use burn::prelude::*;

pub struct RotaryEmbedding<B: Backend> {
    /// Shape: [max_seqlen, head_dim/2, 2, 2] — rotation matrices.
    pub freqs_cis: Tensor<B, 4>,
    pub max_seqlen: usize,
    pub head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Precompute the rotation-matrix table for 1-D positional RoPE.
    ///
    /// Python equivalent:
    ///   freqs = 1.0 / (theta ** (arange(0, dim, 2)[:dim//2] / dim))
    ///   t = arange(max_seqlen)
    ///   freqs = outer(t, freqs)
    ///   cos, sin = freqs.cos(), freqs.sin()
    pub fn new(
        head_dim: usize,
        max_seqlen: usize,
        theta: f64,
        device: &B::Device,
    ) -> Self {
        let half = head_dim / 2;
        let mut table = vec![0f32; max_seqlen * half * 4];

        for pos in 0..max_seqlen {
            for h in 0..half {
                let freq = 1.0 / theta.powf((2 * h) as f64 / head_dim as f64) as f32;
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                let base = (pos * half + h) * 4;
                table[base]     =  c;  // [0,0]
                table[base + 1] = -s;  // [0,1]
                table[base + 2] =  s;  // [1,0]
                table[base + 3] =  c;  // [1,1]
            }
        }

        let freqs_cis = Tensor::<B, 1>::from_data(
            TensorData::new(table, vec![max_seqlen * half * 4]),
            device,
        )
        .reshape([max_seqlen, half, 2, 2]);

        Self { freqs_cis, max_seqlen, head_dim }
    }

    /// Build rotation matrices for sequence positions 0..seq_len.
    /// Returns: [seq_len, head_dim/2, 2, 2]
    pub fn get_freqs(&self, seq_len: usize) -> Tensor<B, 4> {
        assert!(seq_len <= self.max_seqlen,
            "seq_len {seq_len} > max_seqlen {}", self.max_seqlen);
        self.freqs_cis.clone().narrow(0, 0, seq_len)
    }
}

/// Apply RoPE to query and key tensors.
///
/// xq, xk : [B, S, H, D]
/// freqs  : [S, D/2, 2, 2]
///
/// Returns rotated (xq', xk') with the same shape.
pub fn apply_rope<B: Backend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    freqs: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_b, s, h, d] = xq.dims();
    let half = d / 2;

    // cos = freqs[..., 0, 0], sin = freqs[..., 1, 0]
    let cos = freqs
        .clone()
        .narrow(2, 0, 1)
        .narrow(3, 0, 1)
        .reshape([1, s, 1, half]);
    let sin = freqs
        .narrow(2, 1, 1)
        .narrow(3, 0, 1)
        .reshape([1, s, 1, half]);

    (
        rotate_half(xq, cos.clone(), sin.clone(), s, h, half),
        rotate_half(xk, cos, sin, s, h, half),
    )
}

fn rotate_half<B: Backend>(
    x:    Tensor<B, 4>,
    cos:  Tensor<B, 4>,
    sin:  Tensor<B, 4>,
    s:    usize,
    h:    usize,
    half: usize,
) -> Tensor<B, 4> {
    let b = x.dims()[0];
    let pairs = x.reshape([b, s, h, half, 2]);
    let even = pairs.clone().narrow(4, 0, 1).reshape([b, s, h, half]);
    let odd  = pairs.narrow(4, 1, 1).reshape([b, s, h, half]);

    let out_even = even.clone() * cos.clone() - odd.clone() * sin.clone();
    let out_odd  = even * sin + odd * cos;

    Tensor::stack::<5>(vec![out_even, out_odd], 4)
        .reshape([b, s, h, half * 2])
}
