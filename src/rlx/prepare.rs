//! CPU-side token preparation: patch + frequency + channel-location embeddings.
//!
//! FFT and convolutions run on the host; the RLX graph starts at the
//! cross-attention block.

#![allow(clippy::too_many_arguments)]

use rustfft::{num_complex::Complex, FftPlanner};

use super::weights::{ParamBuf, ParamMap};

/// Gather `[B, C, D]` channel-name embeddings from `channel_emb.weight` `[vocab, D]`.
pub fn gather_channel_emb(
    table: &ParamBuf,
    indices: &[i32],
    b: usize,
    c: usize,
    d: usize,
) -> Vec<f32> {
    let vocab = table.shape[0];
    let mut out = vec![0f32; b * c * d];
    for bi in 0..b {
        for ch in 0..c {
            let idx = indices[bi * c + ch] as usize;
            debug_assert!(idx < vocab);
            for j in 0..d {
                out[bi * c * d + ch * d + j] = table.data[idx * d + j];
            }
        }
    }
    out
}

/// Channel-wise z-score along the time axis: `x` is `[C, T]` row-major.
pub fn channel_wise_normalize(x: &mut [f32], c: usize, t: usize) {
    for ch in 0..c {
        let off = ch * t;
        let slice = &mut x[off..off + t];
        let mean: f32 = slice.iter().sum::<f32>() / t as f32;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / t as f32;
        let std = (var + 1e-8).sqrt();
        for v in slice.iter_mut() {
            *v = (*v - mean) / std;
        }
    }
}

fn linear3(x: &[f32], w: &[f32], b: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    let n = x.len() / in_d;
    let mut y = vec![0f32; n * out_d];
    for i in 0..n {
        for o in 0..out_d {
            let mut acc = b[o];
            for j in 0..in_d {
                acc += x[i * in_d + j] * w[j * out_d + o];
            }
            y[i * out_d + o] = acc;
        }
    }
    y
}

/// Matches Burn NdArray `gelu`: `0.5 * x * (1 + erf(x / sqrt(2)))` with `libm::erf` on f64.
fn gelu(x: f32) -> f32 {
    let scaled = (x as f64) * std::f64::consts::FRAC_1_SQRT_2;
    0.5 * x * (1.0 + libm::erf(scaled) as f32)
}

fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let n = x.len() / d;
    let eps = eps as f64;
    let mut y = vec![0f32; x.len()];
    for i in 0..n {
        let slice = &x[i * d..(i + 1) * d];
        let mean: f64 = slice.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var: f64 = slice
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            let normed = ((slice[j] as f64 - mean) * inv) as f32;
            y[i * d + j] = normed * gamma[j] + beta[j];
        }
    }
    y
}

fn nerf_encode(coords: &[f32], n: usize, embed_size: usize) -> Vec<f32> {
    let dim = 3usize;
    let freqs = embed_size / (2 * dim);
    let leftover = embed_size - freqs * 2 * dim;
    let mut out = vec![0f32; n * embed_size];
    for idx in 0..n {
        for f in 0..freqs {
            let band = 2.0_f32.powi(f as i32);
            for axis in 0..dim {
                let v = coords[idx * dim + axis] * band;
                let (s, co) = v.sin_cos();
                let base = idx * embed_size + f * dim * 2 + axis * 2;
                out[base] = s;
                out[base + 1] = co;
            }
        }
        if leftover > 0 {
            let pad_off = idx * embed_size + freqs * dim * 2;
            for j in 0..leftover {
                out[pad_off + j] = 0.0;
            }
        }
    }
    out
}

/// Valid 2-D convolution on NCHW tensors (no implicit padding).
fn conv2d_valid_nchw(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
) -> (usize, usize, Vec<f32>) {
    let h_out = (h - kh) / stride[0] + 1;
    let w_out = (w_in - kw) / stride[1] + 1;
    let mut y = vec![0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let hi = ho * stride[0] + kh_i;
                                let wi = wo * stride[1] + kw_i;
                                let x_idx = ni * c_in * h * w_in + ci * h * w_in + hi * w_in + wi;
                                let w_idx = co * c_in * kh * kw + ci * kh * kw + kh_i * kw + kw_i;
                                acc = f32::mul_add(x[x_idx], w[w_idx], acc);
                            }
                        }
                    }
                    if let Some(bb) = b {
                        acc += bb[co];
                    }
                    let y_idx = ni * c_out * h_out * w_out + co * h_out * w_out + ho * w_out + wo;
                    y[y_idx] = acc;
                }
            }
        }
    }
    (h_out, w_out, y)
}

fn pad_w_symmetric_nchw(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    pad: usize,
) -> (usize, Vec<f32>) {
    let w2 = w + 2 * pad;
    let mut out = vec![0f32; n * c * h * w2];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ni * c * h * w + ci * h * w + hi * w + wi;
                    let dst = ni * c * h * w2 + ci * h * w2 + hi * w2 + pad + wi;
                    out[dst] = x[src];
                }
            }
        }
    }
    (w2, out)
}

fn group_norm_nchw(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    groups: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = x.to_vec();
    let gc = c / groups;
    let eps = eps as f64;
    for ni in 0..n {
        for g in 0..groups {
            let c0 = g * gc;
            let mut mean = 0f64;
            let mut count = 0usize;
            for co in c0..c0 + gc {
                for hi in 0..h {
                    for wi in 0..w {
                        mean += y[ni * c * h * w + co * h * w + hi * w + wi] as f64;
                        count += 1;
                    }
                }
            }
            mean /= count as f64;
            let mut var = 0f64;
            for co in c0..c0 + gc {
                for hi in 0..h {
                    for wi in 0..w {
                        let v = y[ni * c * h * w + co * h * w + hi * w + wi] as f64 - mean;
                        var += v * v;
                    }
                }
            }
            var /= count as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for co in c0..c0 + gc {
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = ni * c * h * w + co * h * w + hi * w + wi;
                        let normed = ((y[idx] as f64 - mean) * inv) as f32;
                        y[idx] = normed * gamma[co] + beta[co];
                    }
                }
            }
        }
    }
    y
}

/// `[B, C, T]` row-major → `[B, 1, C*S, P]` row-major.
fn signal_to_b1csp(signal: &[f32], b: usize, c: usize, t: usize, p: usize) -> (usize, Vec<f32>) {
    let s = t / p;
    let cs = c * s;
    let mut img = vec![0f32; b * cs * p];
    for bi in 0..b {
        for ch in 0..c {
            for pi in 0..s {
                for j in 0..p {
                    img[bi * cs * p + (ch * s + pi) * p + j] =
                        signal[bi * c * t + ch * t + pi * p + j];
                }
            }
        }
    }
    (cs, img)
}

pub fn freq_embed_cpu(signal: &[f32], b: usize, c: usize, t: usize, p: usize, params: &ParamMap) -> Vec<f32> {
    let s = t / p;
    let n_freq = p / 2 + 1;
    let in_f = 2 * n_freq;
    let n_elem = b * c * s;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(p);
    let mut mag_phase = vec![0f32; n_elem * in_f];
    for i in 0..n_elem {
        let mut buf: Vec<Complex<f64>> = signal[i * p..(i + 1) * p]
            .iter()
            .map(|&v| Complex { re: v as f64, im: 0.0 })
            .collect();
        fft.process(&mut buf);
        let off = i * in_f;
        for k in 0..n_freq {
            let re = buf[k].re;
            let im = buf[k].im;
            mag_phase[off + k] = (re * re + im * im).sqrt() as f32;
            mag_phase[off + n_freq + k] = im.atan2(re) as f32; // f64 atan2 → f32, matches Burn FFT path
        }
    }
    let w1 = &params["freq_embed.fc1.weight"].data;
    let b1 = &params["freq_embed.fc1.bias"].data;
    let w2 = &params["freq_embed.fc2.weight"].data;
    let b2 = &params["freq_embed.fc2.bias"].data;
    let hidden = params["freq_embed.fc1.bias"].shape[0];
    let d = params["freq_embed.fc2.bias"].shape[0];
    let h = linear3(&mag_phase, w1, b1, in_f, hidden);
    let h: Vec<f32> = h.into_iter().map(gelu).collect();
    linear3(&h, w2, b2, hidden, d)
}

pub fn patch_embed_cpu(
    signal: &[f32],
    b: usize,
    c: usize,
    t: usize,
    p: usize,
    d: usize,
    params: &ParamMap,
) -> Vec<f32> {
    let (cs, x) = signal_to_b1csp(signal, b, c, t, p);
    let get = |conv: &str| -> (&[f32], &[f32]) {
        (
            &params[&format!("patch_embed.{conv}.weight")].data,
            &params[&format!("patch_embed.{conv}.bias")].data,
        )
    };

    let out_ch = d / 4;
    let kernel = p / 2;
    let w_pad = kernel / 2 - 1;

    // [B, 1, CS, P] with symmetric W padding for conv1 (Burn manual pad).
    let (_, x_padded) = pad_w_symmetric_nchw(&x, b, 1, cs, p, w_pad);
    let (w, bb) = get("conv1");
    let (_, w1, mut h) = conv2d_valid_nchw(
        &x_padded,
        w,
        Some(bb),
        b,
        1,
        cs,
        p + 2 * w_pad,
        out_ch,
        1,
        kernel - 1,
        [1, kernel / 2],
    );
    let (gw, gb) = (
        &params["patch_embed.gn1.weight"].data,
        &params["patch_embed.gn1.bias"].data,
    );
    h = group_norm_nchw(&h, gw, gb, b, out_ch, cs, w1, 4, 1e-5);
    h = h.into_iter().map(gelu).collect();

    // conv2/3: Explicit(0,1) → symmetric W padding of 1, then valid conv.
    let (w, bb) = get("conv2");
    let (_, h_padded) = pad_w_symmetric_nchw(&h, b, out_ch, cs, w1, 1);
    let (_h_h, w2, mut h) = conv2d_valid_nchw(
        &h_padded,
        w,
        Some(bb),
        b,
        out_ch,
        cs,
        w1 + 2,
        out_ch,
        1,
        3,
        [1, 1],
    );
    let (gw, gb) = (
        &params["patch_embed.gn2.weight"].data,
        &params["patch_embed.gn2.bias"].data,
    );
    h = group_norm_nchw(&h, gw, gb, b, out_ch, cs, w2, 4, 1e-5);
    h = h.into_iter().map(gelu).collect();

    let (w, bb) = get("conv3");
    let (_, h_padded) = pad_w_symmetric_nchw(&h, b, out_ch, cs, w2, 1);
    let (_h_h, d_prime, h) = conv2d_valid_nchw(
        &h_padded,
        w,
        Some(bb),
        b,
        out_ch,
        cs,
        w2 + 2,
        out_ch,
        1,
        3,
        [1, 1],
    );
    let (gw, gb) = (
        &params["patch_embed.gn3.weight"].data,
        &params["patch_embed.gn3.bias"].data,
    );
    let h = group_norm_nchw(&h, gw, gb, b, out_ch, cs, d_prime, 4, 1e-5);
    let h: Vec<f32> = h.into_iter().map(gelu).collect();

    // [B, E, CS, D'] → [B, CS, D]  (D' fastest, E slowest — matches Burn swap_dims).
    let e = out_ch;
    let mut out = vec![0f32; b * cs * d];
    for bi in 0..b {
        for i in 0..cs {
            for dp in 0..d_prime {
                for ei in 0..e {
                    let src = bi * e * cs * d_prime + ei * cs * d_prime + i * d_prime + dp;
                    let dst = bi * cs * d + i * d + dp * e + ei;
                    out[dst] = h[src];
                }
            }
        }
    }
    out
}

/// Prepare `x_tokenized` and `decoder_queries` tensors for the RLX graph.
///
/// `signal` must already be channel-normalised when the caller wants normalised
/// inputs (see [`RunEpochOpts`](super::encoder::RunEpochOpts)).
///
/// Returns `(x_tokenized [bt,C,D], decoder_queries [bt,C,D])`.
pub fn prepare_tokens(
    signal: &[f32],
    chan_pos: &[f32],
    channel_emb: Option<&[f32]>,
    b: usize,
    c: usize,
    t: usize,
    patch_size: usize,
    d: usize,
    params: &ParamMap,
) -> (Vec<f32>, Vec<f32>) {
    let p = patch_size;
    let s = t / p;
    let bt = b * s;

    let patch = patch_embed_cpu(signal, b, c, t, p, d, params);
    let freq = freq_embed_cpu(signal, b, c, t, p, params);
    let patched: Vec<f32> = patch
        .iter()
        .zip(freq.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Normalise channel positions to [0,1] per batch item (min/max over channels).
    let mut pos = vec![0f32; b * c * 3];
    for bi in 0..b {
        let off = bi * c * 3;
        let slice = &chan_pos[off..off + c * 3];
        let mut mins = [f32::MAX; 3];
        let mut maxs = [f32::MIN; 3];
        for ch in 0..c {
            for ax in 0..3 {
                let v = slice[ch * 3 + ax];
                mins[ax] = mins[ax].min(v);
                maxs[ax] = maxs[ax].max(v);
            }
        }
        for ch in 0..c {
            for ax in 0..3 {
                let v = slice[ch * 3 + ax];
                let range = maxs[ax] - mins[ax];
                pos[off + ch * 3 + ax] = (v - mins[ax]) / (range + 1e-8);
            }
        }
    }

    let nerf = nerf_encode(&pos, b * c, d);
    let w1 = &params["chan_loc.fc1.weight"].data;
    let b1 = &params["chan_loc.fc1.bias"].data;
    let w2 = &params["chan_loc.fc2.weight"].data;
    let b2 = &params["chan_loc.fc2.bias"].data;
    let gw = &params["chan_loc.norm.weight"].data;
    let gb = &params["chan_loc.norm.bias"].data;
    let hidden = b1.len();
    let mut chan_emb = linear3(&nerf, w1, b1, d, hidden);
    chan_emb = chan_emb.into_iter().map(gelu).collect();
    chan_emb = layer_norm(&chan_emb, gw, gb, hidden, 1e-5);
    let chan_emb = linear3(&chan_emb, w2, b2, hidden, d);

    // x_tokenized: [B*S, C, D] — batch outer, patch inner (matches Burn einops).
    let mut x_tok = vec![0f32; bt * c * d];
    for bi in 0..b {
        for pi in 0..s {
            let bt_i = bi * s + pi;
            for ch in 0..c {
                for j in 0..d {
                    let patch_idx = bi * c * s * d + ch * s * d + pi * d + j;
                    x_tok[bt_i * c * d + ch * d + j] = patched[patch_idx];
                }
            }
        }
    }

    // decoder_queries: Python `.repeat(S, 1, 1)` on [B, C, D] → [S*B, C, D].
    let mut dec_q = vec![0f32; bt * c * d];
    for pi in 0..s {
        for bi in 0..b {
            let bt_i = pi * b + bi;
            for ch in 0..c {
                for j in 0..d {
                    dec_q[bt_i * c * d + ch * d + j] = chan_emb[bi * c * d + ch * d + j];
                }
            }
        }
    }

    if let Some(emb) = channel_emb {
        for pi in 0..s {
            for bi in 0..b {
                let bt_i = pi * b + bi;
                for ch in 0..c {
                    for j in 0..d {
                        let idx = bt_i * c * d + ch * d + j;
                        dec_q[idx] += emb[bi * c * d + ch * d + j];
                    }
                }
            }
        }
    }

    // Add channel location embedding to x_tok (Python repeat layout).
    for pi in 0..s {
        for bi in 0..b {
            let bt_i = pi * b + bi;
            for ch in 0..c {
                for j in 0..d {
                    let idx = bt_i * c * d + ch * d + j;
                    x_tok[idx] += chan_emb[bi * c * d + ch * d + j];
                }
            }
        }
    }

    (x_tok, dec_q)
}
