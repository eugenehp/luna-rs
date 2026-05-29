//! LUNA forward pass (cross-attention → rotary encoder → reconstruction head).

#![allow(clippy::too_many_arguments)]

use rlx::ir::GraphExt;
use rlx::ops::MaskKind;
use rlx::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ForwardSpec {
    pub b: usize,
    pub c: usize,
    pub s: usize,
    pub bt: usize,
    pub d: usize,
    pub q: usize,
    pub hidden: usize,
    pub nh_ca: usize,
    pub nh_rot: usize,
    pub dh_ca: usize,
    pub dh_rot: usize,
    pub depth: usize,
    pub ff_ca: usize,
    pub ff_rot: usize,
    pub patch_size: usize,
    pub norm_eps: f32,
    /// `0` = reconstruction head, `>0` = classification logits.
    pub num_classes: usize,
    /// Classifier MHA head count (`ModelConfig::num_heads`).
    pub nh_cls: usize,
}

fn s1(d: usize) -> Shape {
    Shape::new(&[d], DType::F32)
}
fn s2(a: usize, b: usize) -> Shape {
    Shape::new(&[a, b], DType::F32)
}
fn s3(a: usize, b: usize, c: usize) -> Shape {
    Shape::new(&[a, b, c], DType::F32)
}
fn s4(a: usize, b: usize, c: usize, d: usize) -> Shape {
    Shape::new(&[a, b, c, d], DType::F32)
}

/// `[B,S,H,D]` → `[B,H,S,D]` for MLX SDPA (CPU/Metal accept both).
fn attn_bshd_to_bhsd(g: &mut Graph, t: NodeId) -> NodeId {
    g.transpose_(t, vec![0, 2, 1, 3])
}

fn ln(g: &mut Graph, x: NodeId, w: NodeId, b: NodeId, eps: f32) -> NodeId {
    g.ln(x, w, b, eps)
}

fn bias_add(g: &mut Graph, x: NodeId, b: NodeId) -> NodeId {
    g.add(x, b)
}

fn rotate_half(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    b: usize,
    s: usize,
    h: usize,
    d: usize,
) -> NodeId {
    let half = d / 2;
    let pairs = g.reshape_(x, vec![b as i64, s as i64, h as i64, half as i64, 2]);
    let even5 = g.narrow_(pairs, 4, 0, 1);
    let odd5 = g.narrow_(pairs, 4, 1, 1);
    let even = g.reshape_(even5, vec![b as i64, s as i64, h as i64, half as i64]);
    let odd = g.reshape_(odd5, vec![b as i64, s as i64, h as i64, half as i64]);
    let ec = g.mul(even, cos);
    let os = g.mul(odd, sin);
    let out_even = g.sub(ec, os);
    let es = g.mul(even, sin);
    let oc = g.mul(odd, cos);
    let out_odd = g.add(es, oc);
    let e5 = g.reshape_(out_even, vec![b as i64, s as i64, h as i64, half as i64, 1]);
    let o5 = g.reshape_(out_odd, vec![b as i64, s as i64, h as i64, half as i64, 1]);
    let stacked = g.concat_(vec![e5, o5], 4);
    g.reshape_(stacked, vec![b as i64, s as i64, h as i64, d as i64])
}

fn mha(
    g: &mut Graph,
    q_in: NodeId,
    k_in: NodeId,
    v_in: NodeId,
    prefix: &str,
    batch: usize,
    s_q: usize,
    s_kv: usize,
    d: usize,
    nh: usize,
    dh: usize,
) -> NodeId {
    let h_total = nh * dh;
    let wq = g.param(format!("{prefix}.wq.weight"), s2(d, h_total));
    let wk = g.param(format!("{prefix}.wk.weight"), s2(d, h_total));
    let wv = g.param(format!("{prefix}.wv.weight"), s2(d, h_total));
    let wo = g.param(format!("{prefix}.wo.weight"), s2(h_total, d));
    let wq_b = g.param(format!("{prefix}.wq.bias"), s1(h_total));
    let wk_b = g.param(format!("{prefix}.wk.bias"), s1(h_total));
    let wv_b = g.param(format!("{prefix}.wv.bias"), s1(h_total));
    let wo_b = g.param(format!("{prefix}.wo.bias"), s1(d));

    let qm = g.mm(q_in, wq);
    let q = bias_add(g, qm, wq_b);
    let km = g.mm(k_in, wk);
    let k = bias_add(g, km, wk_b);
    let vm = g.mm(v_in, wv);
    let v = bias_add(g, vm, wv_b);

    let q4 = g.reshape_(q, vec![batch as i64, s_q as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![batch as i64, s_kv as i64, nh as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![batch as i64, s_kv as i64, nh as i64, dh as i64]);

    let q_bhsd = attn_bshd_to_bhsd(g, q4);
    let k_bhsd = attn_bshd_to_bhsd(g, k4);
    let v_bhsd = attn_bshd_to_bhsd(g, v4);
    let attn = g.attention_kind(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        nh,
        dh,
        MaskKind::None,
        s4(batch, nh, s_q, dh),
    );
    let attn_bshd = attn_bshd_to_bhsd(g, attn);
    let attn_3 = g.reshape_(attn_bshd, vec![batch as i64, s_q as i64, h_total as i64]);
    let om = g.mm(attn_3, wo);
    bias_add(g, om, wo_b)
}

fn transformer_encoder_layer(
    g: &mut Graph,
    x: NodeId,
    prefix: &str,
    batch: usize,
    s: usize,
    d: usize,
    nh: usize,
    dh: usize,
    ff: usize,
    eps: f32,
) -> NodeId {
    let n1w = g.param(format!("{prefix}.norm1.weight"), s1(d));
    let n1b = g.param(format!("{prefix}.norm1.bias"), s1(d));
    let normed = ln(g, x, n1w, n1b, eps);
    let attn = mha(g, normed, normed, normed, &format!("{prefix}.self_attn"), batch, s, s, d, nh, dh);
    let x = g.add(x, attn);

    let n2w = g.param(format!("{prefix}.norm2.weight"), s1(d));
    let n2b = g.param(format!("{prefix}.norm2.bias"), s1(d));
    let normed = ln(g, x, n2w, n2b, eps);
    let l1w = g.param(format!("{prefix}.linear1.weight"), s2(d, ff));
    let l1b = g.param(format!("{prefix}.linear1.bias"), s1(ff));
    let l2w = g.param(format!("{prefix}.linear2.weight"), s2(ff, d));
    let l2b = g.param(format!("{prefix}.linear2.bias"), s1(d));
    let l1m = g.mm(normed, l1w);
    let l1b_ = bias_add(g, l1m, l1b);
    let h = g.gelu(l1b_);
    let l2m = g.mm(h, l2w);
    let ff_out = bias_add(g, l2m, l2b);
    g.add(x, ff_out)
}

fn cross_attention_block(g: &mut Graph, x: NodeId, queries: NodeId, spec: &ForwardSpec) -> NodeId {
    let d = spec.d;
    let qn = spec.q;
    let bt = spec.bt;
    let c = spec.c;
    let nh = spec.nh_ca;
    let dh = spec.dh_ca;
    let eps = spec.norm_eps;

    let qnw = g.param("cross_attn.queries_norm.weight", s1(d));
    let qnb = g.param("cross_attn.queries_norm.bias", s1(d));
    let knw = g.param("cross_attn.keys_norm.weight", s1(d));
    let knb = g.param("cross_attn.keys_norm.bias", s1(d));
    let vnw = g.param("cross_attn.values_norm.weight", s1(d));
    let vnb = g.param("cross_attn.values_norm.bias", s1(d));

    let q = ln(g, queries, qnw, qnb, eps);
    let k = ln(g, x, knw, knb, eps);
    let v = ln(g, x, vnw, vnb, eps);

    let mut out = mha(
        g,
        q,
        k,
        v,
        "cross_attn.cross_attention",
        bt,
        qn,
        c,
        d,
        nh,
        dh,
    );

    let f1w = g.param("cross_attn.ffn.fc1.weight", s2(d, spec.ff_ca));
    let f1b = g.param("cross_attn.ffn.fc1.bias", s1(spec.ff_ca));
    let fnw = g.param("cross_attn.ffn.norm.weight", s1(spec.ff_ca));
    let fnb = g.param("cross_attn.ffn.norm.bias", s1(spec.ff_ca));
    let f2w = g.param("cross_attn.ffn.fc2.weight", s2(spec.ff_ca, d));
    let f2b = g.param("cross_attn.ffn.fc2.bias", s1(d));

    let residual = out;
    let f1m = g.mm(out, f1w);
    let f1h = bias_add(g, f1m, f1b);
    let h = g.gelu(f1h);
    let h = ln(g, h, fnw, fnb, eps);
    let f2m = g.mm(h, f2w);
    let f2h = bias_add(g, f2m, f2b);
    out = g.add(residual, f2h);

    for i in 0..3 {
        out = transformer_encoder_layer(
            g,
            out,
            &format!("cross_attn.query_self_attn.layers.{i}"),
            bt,
            qn,
            d,
            nh,
            dh,
            spec.ff_ca,
            eps,
        );
    }
    out
}

fn rotary_block(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    spec: &ForwardSpec,
    idx: usize,
) -> NodeId {
    let p = format!("blocks.{idx}");
    let d = spec.hidden;
    let nh = spec.nh_rot;
    let dh = spec.dh_rot;
    let b = spec.b;
    let s = spec.s;
    let eps = spec.norm_eps;

    let n1w = g.param(format!("{p}.norm1.weight"), s1(d));
    let n1b = g.param(format!("{p}.norm1.bias"), s1(d));
    let xn = ln(g, x, n1w, n1b, eps);

    let wq = g.param(format!("{p}.attn.wq.weight"), s2(d, nh * dh));
    let wk = g.param(format!("{p}.attn.wk.weight"), s2(d, nh * dh));
    let wv = g.param(format!("{p}.attn.wv.weight"), s2(d, nh * dh));
    let wo = g.param(format!("{p}.attn.proj.weight"), s2(nh * dh, d));
    let wq_b = g.param(format!("{p}.attn.wq.bias"), s1(nh * dh));
    let wk_b = g.param(format!("{p}.attn.wk.bias"), s1(nh * dh));
    let wv_b = g.param(format!("{p}.attn.wv.bias"), s1(nh * dh));
    let wo_b = g.param(format!("{p}.attn.proj.bias"), s1(d));

    let qm = g.mm(xn, wq);
    let q = bias_add(g, qm, wq_b);
    let km = g.mm(xn, wk);
    let k = bias_add(g, km, wk_b);
    let vm = g.mm(xn, wv);
    let v = bias_add(g, vm, wv_b);

    let q4 = g.reshape_(q, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![b as i64, s as i64, nh as i64, dh as i64]);

    let q_rot = rotate_half(g, q4, cos, sin, b, s, nh, dh);
    let k_rot = rotate_half(g, k4, cos, sin, b, s, nh, dh);

    let q_bhsd = attn_bshd_to_bhsd(g, q_rot);
    let k_bhsd = attn_bshd_to_bhsd(g, k_rot);
    let v_bhsd = attn_bshd_to_bhsd(g, v4);
    let attn = g.attention_kind(q_bhsd, k_bhsd, v_bhsd, nh, dh, MaskKind::None, s4(b, nh, s, dh));
    let attn_bshd = attn_bshd_to_bhsd(g, attn);
    let attn_3 = g.reshape_(attn_bshd, vec![b as i64, s as i64, (nh * dh) as i64]);
    let aom = g.mm(attn_3, wo);
    let attn_out = bias_add(g, aom, wo_b);
    let x = g.add(x, attn_out);

    let n2w = g.param(format!("{p}.norm2.weight"), s1(d));
    let n2b = g.param(format!("{p}.norm2.bias"), s1(d));
    let hn = ln(g, x, n2w, n2b, eps);

    let fc1w = g.param(format!("{p}.mlp.fc1.weight"), s2(d, spec.ff_rot));
    let fc1b = g.param(format!("{p}.mlp.fc1.bias"), s1(spec.ff_rot));
    let mnw = g.param(format!("{p}.mlp.norm.weight"), s1(spec.ff_rot));
    let mnb = g.param(format!("{p}.mlp.norm.bias"), s1(spec.ff_rot));
    let fc2w = g.param(format!("{p}.mlp.fc2.weight"), s2(spec.ff_rot, d));
    let fc2b = g.param(format!("{p}.mlp.fc2.bias"), s1(d));

    let m1 = g.mm(hn, fc1w);
    let m1b = bias_add(g, m1, fc1b);
    let h = g.gelu(m1b);
    let h = ln(g, h, mnw, mnb, eps);
    let m2 = g.mm(h, fc2w);
    let m2b = bias_add(g, m2, fc2b);
    g.add(x, m2b)
}

fn reconstruction_head(g: &mut Graph, enc: NodeId, tgt: NodeId, spec: &ForwardSpec) -> NodeId {
    let d = spec.d;
    let qn = spec.q;
    let b = spec.b;
    let s = spec.s;
    let bt = spec.bt;
    let c = spec.c;
    let p = spec.patch_size;
    let nh = spec.nh_ca;
    let dh = spec.dh_ca;
    let eps = spec.norm_eps;
    let dp = "decoder_head.decoder_pred.layers.0";

    let memory = g.reshape_(enc, vec![bt as i64, qn as i64, d as i64]);
    let mut x = tgt;

    let n1w = g.param(format!("{dp}.norm1.weight"), s1(d));
    let n1b = g.param(format!("{dp}.norm1.bias"), s1(d));
    let normed = ln(g, x, n1w, n1b, eps);
    let sa = mha(
        g,
        normed,
        normed,
        normed,
        &format!("{dp}.self_attn"),
        bt,
        c,
        c,
        d,
        nh,
        dh,
    );
    x = g.add(x, sa);

    let n2w = g.param(format!("{dp}.norm2.weight"), s1(d));
    let n2b = g.param(format!("{dp}.norm2.bias"), s1(d));
    let normed = ln(g, x, n2w, n2b, eps);
    let ca = mha(
        g,
        normed,
        memory,
        memory,
        &format!("{dp}.multihead_attn"),
        bt,
        c,
        qn,
        d,
        nh,
        dh,
    );
    x = g.add(x, ca);

    let n3w = g.param(format!("{dp}.norm3.weight"), s1(d));
    let n3b = g.param(format!("{dp}.norm3.bias"), s1(d));
    let normed = ln(g, x, n3w, n3b, eps);
    let l1w = g.param(format!("{dp}.linear1.weight"), s2(d, d * 4));
    let l1b = g.param(format!("{dp}.linear1.bias"), s1(d * 4));
    let l2w = g.param(format!("{dp}.linear2.weight"), s2(d * 4, d));
    let l2b = g.param(format!("{dp}.linear2.bias"), s1(d));
    let l1m = g.mm(normed, l1w);
    let l1b_ = bias_add(g, l1m, l1b);
    let ff = g.gelu(l1b_);
    let l2m = g.mm(ff, l2w);
    let l2b_ = bias_add(g, l2m, l2b);
    x = g.add(x, l2b_);

    let onw = g.param("decoder_head.norm.weight", s1(d));
    let onb = g.param("decoder_head.norm.bias", s1(d));
    x = ln(g, x, onw, onb, eps);

    let of1w = g.param("decoder_head.decoder_linear.fc1.weight", s2(d, d * 4));
    let of1b = g.param("decoder_head.decoder_linear.fc1.bias", s1(d * 4));
    let of2w = g.param("decoder_head.decoder_linear.fc2.weight", s2(d * 4, p));
    let of2b = g.param("decoder_head.decoder_linear.fc2.bias", s1(p));

    let o1m = g.mm(x, of1w);
    let o1b_ = bias_add(g, o1m, of1b);
    let h = g.gelu(o1b_);
    let o2m = g.mm(h, of2w);
    let out = bias_add(g, o2m, of2b);

    // [bt, C, P] → [B, C, T]  (matches Burn: reshape [B,t,C,P] → swap_dims(1,2) → flatten)
    let out5 = g.reshape_(out, vec![b as i64, s as i64, c as i64, p as i64]);
    let out5 = g.transpose_(out5, vec![0, 2, 1, 3]);
    g.reshape_(out5, vec![b as i64, c as i64, (s * p) as i64])
}

fn classification_head(g: &mut Graph, enc: NodeId, agg: NodeId, spec: &ForwardSpec) -> NodeId {
    let hidden = spec.hidden;
    let b = spec.b;
    let nh = spec.nh_cls;
    let dh = hidden / nh;
    let nc = spec.num_classes;

    let ca = mha(
        g,
        agg,
        enc,
        enc,
        "classifier.decoder_attn",
        b,
        1,
        spec.s,
        hidden,
        nh,
        dh,
    );
    let logits_in = g.reshape_(ca, vec![b as i64, hidden as i64]);
    let f1w = g.param("classifier.decoder_ffn.fc1.weight", s2(hidden, hidden * 4));
    let f1b = g.param("classifier.decoder_ffn.fc1.bias", s1(hidden * 4));
    let f2w = g.param("classifier.decoder_ffn.fc2.weight", s2(hidden * 4, nc));
    let f2b = g.param("classifier.decoder_ffn.fc2.bias", s1(nc));
    let l1m = g.mm(logits_in, f1w);
    let l1b_ = bias_add(g, l1m, f1b);
    let h = g.gelu(l1b_);
    let l2m = g.mm(h, f2w);
    bias_add(g, l2m, f2b)
}

/// Build the LUNA forward graph for one `(batch, channels, time)` shape.
///
/// Inputs:
/// * `x_tokenized` — `[B*S, C, D]`
/// * `queries` — `[B*S, Q, D]` (expanded cross-attn query prototypes)
/// * `decoder_queries` — `[B*S, C, D]` (reconstruction only)
/// * `agg_query` — `[B, 1, Q*D]` (classification only)
/// * `freqs_cos`, `freqs_sin` — `[1, S, 1, head_dim/2]`
///
/// Output: `[B, C, T]` reconstruction or `[B, num_classes]` logits.
pub fn build_forward_graph(spec: &ForwardSpec) -> Graph {
    let mut g = Graph::new("luna_forward");

    let x = g.input("x_tokenized", s3(spec.bt, spec.c, spec.d));
    let queries = g.input("queries", s3(spec.bt, spec.q, spec.d));
    let cos = g.input("freqs_cos", s4(1, spec.s, 1, spec.dh_rot / 2));
    let sin = g.input("freqs_sin", s4(1, spec.s, 1, spec.dh_rot / 2));

    let unified = cross_attention_block(&mut g, x, queries, spec);
    let enc = g.reshape_(unified, vec![spec.b as i64, spec.s as i64, spec.hidden as i64]);

    let mut h = enc;
    for i in 0..spec.depth {
        h = rotary_block(&mut g, h, cos, sin, spec, i);
    }

    let nw = g.param("norm.weight", s1(spec.hidden));
    let nb = g.param("norm.bias", s1(spec.hidden));
    let h = ln(&mut g, h, nw, nb, spec.norm_eps);

    let out = if spec.num_classes > 0 {
        let agg = g.input("agg_query", s3(spec.b, 1, spec.hidden));
        classification_head(&mut g, h, agg, spec)
    } else {
        let dec_q = g.input("decoder_queries", s3(spec.bt, spec.c, spec.d));
        reconstruction_head(&mut g, h, dec_q, spec)
    };
    g.set_outputs(vec![out]);
    g
}
