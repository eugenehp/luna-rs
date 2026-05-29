//! IR-level smoke test for the RLX LUNA forward graph.

use luna_rs::rlx::graph::{build_forward_graph, ForwardSpec};
use luna_rs::rlx::rope_helpers::{build_rope_table, precompute_rope};

fn zero_fill_params(graph: &rlx::Graph, compiled: &mut rlx::CompiledGraph) {
    use rlx::Op;
    for node in graph.nodes() {
        let Op::Param { name } = &node.op else {
            continue;
        };
        let n = node.shape.num_elements().expect("param shape must be static");
        compiled.set_param(name, &vec![0.0; n]);
    }
}

#[test]
fn forward_graph_compiles_and_runs() {
    let spec = ForwardSpec {
        b: 1,
        c: 4,
        s: 8,
        bt: 8,
        d: 16,
        q: 2,
        hidden: 32,
        nh_ca: 2,
        nh_rot: 4,
        dh_ca: 8,
        dh_rot: 8,
        depth: 2,
        ff_ca: 64,
        ff_rot: 128,
        patch_size: 40,
        norm_eps: 1e-5,
        num_classes: 0,
        nh_cls: 2,
    };

    let graph = build_forward_graph(&spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());
    zero_fill_params(&graph, &mut compiled);

    let x = vec![0.1_f32; spec.bt * spec.c * spec.d];
    let q = vec![0.2_f32; spec.bt * spec.q * spec.d];
    let dq = vec![0.3_f32; spec.bt * spec.c * spec.d];
    let table = build_rope_table(spec.dh_rot * 2, 64, 10_000.0);
    let (cos, sin) = precompute_rope(&table, spec.dh_rot * 2, spec.s);

    let outs = compiled.run(&[
        ("x_tokenized", &x),
        ("queries", &q),
        ("decoder_queries", &dq),
        ("freqs_cos", &cos),
        ("freqs_sin", &sin),
    ]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), spec.b * spec.c * spec.s * spec.patch_size);
}

#[test]
fn classification_graph_compiles_and_runs() {
    let spec = ForwardSpec {
        b: 1,
        c: 4,
        s: 8,
        bt: 8,
        d: 16,
        q: 2,
        hidden: 32,
        nh_ca: 2,
        nh_rot: 4,
        dh_ca: 8,
        dh_rot: 8,
        depth: 2,
        ff_ca: 64,
        ff_rot: 128,
        patch_size: 40,
        norm_eps: 1e-5,
        num_classes: 3,
        nh_cls: 2,
    };

    let graph = build_forward_graph(&spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());
    zero_fill_params(&graph, &mut compiled);

    let x = vec![0.1_f32; spec.bt * spec.c * spec.d];
    let q = vec![0.2_f32; spec.bt * spec.q * spec.d];
    let agg = vec![0.3_f32; spec.b * spec.hidden];
    let cos = vec![1.0_f32; spec.s * spec.dh_rot / 2];
    let sin = vec![0.0_f32; spec.s * spec.dh_rot / 2];
    let outs = compiled.run(&[
        ("x_tokenized", &x),
        ("queries", &q),
        ("agg_query", &agg),
        ("freqs_cos", &cos),
        ("freqs_sin", &sin),
    ]);
    assert_eq!(outs[0].len(), spec.b * spec.num_classes);
}
