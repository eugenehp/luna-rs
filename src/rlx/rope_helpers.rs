//! CPU-side 1-D RoPE tables for LUNA's temporal encoder.
//!
//! LUNA uses standard 1-D RoPE (not ZUNA's 4-D axial layout). The `cos` /
//! `sin` tensors are fed into the RLX graph as inputs, matching the Burn
//! `RotaryEmbedding::get_freqs` layout.

/// Build `[max_seqlen, half, 2, 2]` rotation matrices (row-major flatten).
pub fn build_rope_table(head_dim: usize, max_seqlen: usize, theta: f64) -> Vec<f32> {
    let half = head_dim / 2;
    let mut table = vec![0f32; max_seqlen * half * 4];
    for pos in 0..max_seqlen {
        for h in 0..half {
            // Match `RotaryEmbedding::new` in `src/model/rope.rs` (f32 angles).
            let freq = (1.0 / theta.powf((2 * h) as f64 / head_dim as f64)) as f32;
            let angle = pos as f32 * freq;
            let (s, c) = angle.sin_cos();
            let base = (pos * half + h) * 4;
            table[base] = c;
            table[base + 1] = -s;
            table[base + 2] = s;
            table[base + 3] = c;
        }
    }
    table
}

/// `cos` / `sin` for positions `0..seq_len`, each `[1, seq_len, 1, half]`.
pub fn precompute_rope(
    rope_table: &[f32],
    head_dim: usize,
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; seq_len * half];
    let mut sin = vec![0f32; seq_len * half];
    for pos in 0..seq_len {
        for h in 0..half {
            let base = (pos * half + h) * 4;
            cos[pos * half + h] = rope_table[base];
            sin[pos * half + h] = -rope_table[base + 1];
        }
    }
    (cos, sin)
}
