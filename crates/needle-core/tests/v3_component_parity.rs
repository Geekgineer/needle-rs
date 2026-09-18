//! Needle 3 component parity, against upstream's own Flax modules.
//!
//! Needs `tests/v3_component_vectors.json` + `.f32` from
//! `tools/gen_v3_component_parity.py`. Each component is checked on its own,
//! so a sign or transpose error reads as "HadamardMLP is wrong" rather than
//! surfacing twenty layers later as wrong logits.

use std::path::Path;

use needle_core::ops::sigmoid;
use needle_core::v3::attention::{attend, causal_depthwise_conv, norm_and_rope, AttnDims, KvStore};
use needle_core::v3::engram::{engram_indices, ngram_valid, value_conv, EngramDims};
use needle_core::v3::heads::{ProbeHead, ProbePool};
use needle_core::v3::kernels::{hada_blocks, hadamard_mlp, HadaMlp, HadaPerms};

/// `y = x · W` for a row-major `(in, out)` kernel, the orientation the
/// rebuilt parameter tree stores.
fn dense(x: &[f32], w: &[f32], n_in: usize, n_out: usize, y: &mut [f32]) {
    y.fill(0.0);
    for (i, &xi) in x.iter().enumerate().take(n_in) {
        if xi == 0.0 {
            continue;
        }
        let row = &w[i * n_out..(i + 1) * n_out];
        for (yo, &wo) in y.iter_mut().zip(row) {
            *yo += xi * wo;
        }
    }
}

const JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_component_vectors.json"
);
const F32: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_component_vectors.f32"
);

struct Fixture {
    meta: serde_json::Value,
    floats: Vec<f32>,
}

impl Fixture {
    fn load() -> Option<Self> {
        if !Path::new(JSON).exists() || !Path::new(F32).exists() {
            println!(
                "skipping v3 component parity: run\n  \
                 JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
                 tools/gen_v3_component_parity.py"
            );
            return None;
        }
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(JSON).unwrap()).unwrap();
        let raw = std::fs::read(F32).unwrap();
        let floats = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self { meta, floats })
    }

    fn get(&self, name: &str) -> Vec<f32> {
        let e = &self.meta["components"][name];
        let off = e["offset"].as_u64().unwrap() as usize;
        let len = e["len"].as_u64().unwrap() as usize;
        self.floats[off..off + len].to_vec()
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.meta["components"][name]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }
}

#[test]
fn hadamard_mlp_matches_the_reference() {
    let Some(f) = Fixture::load() else { return };

    let d_model = f.meta["geometry"]["d_model"].as_u64().unwrap() as usize;
    let hada_n = f.meta["geometry"]["hada_n"].as_u64().unwrap() as usize;
    let cond_rank = f.meta["geometry"]["cond_rank"].as_u64().unwrap() as usize;
    assert_eq!(hada_blocks(hada_n), (32, 32), "block sizes");

    let perms = HadaPerms {
        p1: f.get("hada_p1").iter().map(|&v| v as u32).collect(),
        p2: f.get("hada_p2").iter().map(|&v| v as u32).collect(),
    };
    let w = HadaMlp {
        d1: f.get("mlp_d1"),
        d2: f.get("mlp_d2"),
        b2: f.get("mlp_b2"),
        d3: f.get("mlp_d3"),
        d4: f.get("mlp_d4"),
        w1: (f.get("mlp_w1a"), f.get("mlp_w1b")),
        w2: (f.get("mlp_w2a"), f.get("mlp_w2b")),
        w3: (f.get("mlp_w3a"), f.get("mlp_w3b")),
        cond_v: f.get("mlp_cond_v"),
        cond_u: f.get("mlp_cond_u"),
        cond_rank,
    };

    let x_all = f.get("mlp_in");
    let want_all = f.get("mlp_out");
    let seq = f.shape("mlp_in")[0];
    assert_eq!(x_all.len(), seq * d_model);

    // Deviation is measured against the output RMS, not per element. A
    // per-element relative metric explodes on near-zero outputs and says
    // nothing useful; the fixture is generated in float32 precisely so this
    // number reflects the arithmetic rather than the reference's dtype.
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    let mut sq = 0.0f64;
    for t in 0..seq {
        let x = &x_all[t * d_model..(t + 1) * d_model];
        let want = &want_all[t * d_model..(t + 1) * d_model];
        let mut got = vec![0.0f32; d_model];
        hadamard_mlp(x, &w, &perms, hada_n, &mut got);

        for (i, (&g, &e)) in got.iter().zip(want).enumerate() {
            sq += (e as f64) * (e as f64);
            let d = (g - e).abs();
            if d > worst {
                worst = d;
                worst_at = (t, i);
            }
        }
    }
    let rms = (sq / (seq * d_model) as f64).sqrt() as f32;
    let rel = worst / rms;

    println!(
        "HadamardMLP: max abs deviation {worst:.3e} against output RMS {rms:.1} \
         = {rel:.3e} relative (worst at position {} dim {})",
        worst_at.0, worst_at.1
    );
    assert!(
        rel < 1e-4,
        "HadamardMLP deviates from the f32 reference by {rel:.3e} of RMS \
         (abs {worst:.3e}) at position {} dim {}",
        worst_at.0,
        worst_at.1
    );
}

#[test]
fn attention_matches_the_reference() {
    let Some(f) = Fixture::load() else { return };
    let g = &f.meta["geometry"];
    let u = |k: &str| g[k].as_u64().unwrap() as usize;

    let (d_model, seq) = (u("d_model"), u("seq"));
    let d = AttnDims {
        seq,
        num_heads: u("num_heads"),
        num_kv_heads: u("num_kv_heads"),
        qk_head_dim: u("qk_head_dim"),
        v_head_dim: u("v_head_dim"),
    };
    let taps_n = u("qkv_conv_taps");
    let window = u("sliding_window");

    let x = f.get("attn_in");
    let want = f.get("attn_out");
    let (q_w, k_w, v_w) = (
        f.get("attn_q_proj"),
        f.get("attn_k_proj"),
        f.get("attn_v_proj"),
    );
    let (gate_w, out_w) = (f.get("attn_gate_proj"), f.get("attn_out_proj"));
    let (q_s, k_s) = (f.get("attn_q_norm"), f.get("attn_k_norm"));
    let (cos, sin) = (f.get("attn_rope_cos"), f.get("attn_rope_sin"));

    let q_dim = d.num_heads * d.qk_head_dim;
    let k_dim = d.num_kv_heads * d.qk_head_dim;
    let v_dim = d.num_kv_heads * d.v_head_dim;
    let o_dim = d.num_heads * d.v_head_dim;

    // Projections. In production these are packed CqWeight matvecs; the
    // kernels under test are the same either way.
    let mut q = vec![0.0f32; seq * q_dim];
    let mut k = vec![0.0f32; seq * k_dim];
    let mut v = vec![0.0f32; seq * v_dim];
    for t in 0..seq {
        let xt = &x[t * d_model..(t + 1) * d_model];
        dense(xt, &q_w, d_model, q_dim, &mut q[t * q_dim..(t + 1) * q_dim]);
        dense(xt, &k_w, d_model, k_dim, &mut k[t * k_dim..(t + 1) * k_dim]);
        dense(xt, &v_w, d_model, v_dim, &mut v[t * v_dim..(t + 1) * v_dim]);
    }

    if taps_n > 0 {
        causal_depthwise_conv(&mut q, &f.get("attn_q_taps"), seq, q_dim, taps_n);
        causal_depthwise_conv(&mut k, &f.get("attn_k_taps"), seq, k_dim, taps_n);
        causal_depthwise_conv(&mut v, &f.get("attn_v_taps"), seq, v_dim, taps_n);
    }

    norm_and_rope(&mut q, &q_s, &cos, &sin, seq, d.num_heads, d.qk_head_dim);
    norm_and_rope(&mut k, &k_s, &cos, &sin, seq, d.num_kv_heads, d.qk_head_dim);

    let mut attn = vec![0.0f32; seq * o_dim];
    let span = if window == 0 { None } else { Some(window) };
    attend(&q, KvStore::F32 { k: &k, v: &v }, d, span, &mut attn);

    // out = (attn ⊙ sigmoid(x · gate_proj)) · out_proj
    let mut got = vec![0.0f32; seq * d_model];
    let mut gate = vec![0.0f32; o_dim];
    for t in 0..seq {
        let xt = &x[t * d_model..(t + 1) * d_model];
        dense(xt, &gate_w, d_model, o_dim, &mut gate);
        let a = &mut attn[t * o_dim..(t + 1) * o_dim];
        for (ai, &gi) in a.iter_mut().zip(gate.iter()) {
            *ai *= sigmoid(gi);
        }
        dense(
            a,
            &out_w,
            o_dim,
            d_model,
            &mut got[t * d_model..(t + 1) * d_model],
        );
    }

    let mut worst = 0.0f32;
    let mut at = 0usize;
    let mut sq = 0.0f64;
    for (i, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
        sq += (wv as f64) * (wv as f64);
        let dd = (gv - wv).abs();
        if dd > worst {
            worst = dd;
            at = i;
        }
    }
    let rms = (sq / want.len() as f64).sqrt() as f32;
    let rel = worst / rms;
    println!(
        "attention: max abs deviation {worst:.3e} against output RMS {rms:.3} \
         = {rel:.3e} relative (worst at flat index {at})"
    );
    assert!(
        rel < 1e-4,
        "attention deviates from the f32 reference by {rel:.3e} of RMS (abs {worst:.3e})"
    );
}

#[test]
fn engram_hash_indices_are_exact() {
    let Some(f) = Fixture::load() else { return };
    let g = &f.meta["geometry"];
    let u = |k: &str| g[k].as_u64().unwrap() as usize;

    let orders: Vec<usize> = g["engram_orders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let heads = u("engram_heads");
    let slots = u("engram_slots") as u32;
    let seed_heads = u("engram_seed_heads");
    let num_tables = u("engram_num_tables");

    let tokens: Vec<u32> = f.meta["engram_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let got = engram_indices(&tokens, &orders, heads, slots, seed_heads);
    let want = f.meta["engram_indices"].as_array().unwrap();
    assert_eq!(want.len(), tokens.len(), "one row per position");

    for (p, row) in want.iter().enumerate() {
        let row = row.as_array().unwrap();
        assert_eq!(row.len(), num_tables);
        for (t, cell) in row.iter().enumerate() {
            let expect = cell.as_u64().unwrap() as u32;
            assert_eq!(
                got[p * num_tables + t],
                expect,
                "slot for position {p} table {t}"
            );
        }
    }
    println!(
        "engram: {} positions x {num_tables} tables hash exactly",
        tokens.len()
    );
}

#[test]
fn engram_keys_and_values_match_the_reference() {
    let Some(f) = Fixture::load() else { return };
    let g = &f.meta["geometry"];
    let u = |k: &str| g[k].as_u64().unwrap() as usize;

    let d_model = u("d_model");
    let seq = u("engram_seq");
    let orders: Vec<usize> = g["engram_orders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let heads = u("engram_heads");
    let dims = EngramDims {
        num_tables: u("engram_num_tables"),
        slots: u("engram_slots"),
        sub_dim: u("engram_sub_dim"),
        d_model,
        conv_taps: 4,
        conv_dilation: u("engram_conv_dilation"),
        max_order: *orders.iter().max().unwrap(),
    };

    // Rows already gathered by the fixture; masking is what is under test.
    let rows = f.get("engram_rows");
    let fetch_dim = dims.num_tables * dims.sub_dim;
    assert_eq!(
        fetch_dim, d_model,
        "concatenated fetch should be d_model wide"
    );

    let key_w = f.get("engram_key_proj");
    let val_w = f.get("engram_value_proj");
    let taps = f.get("engram_taps");

    let mut k_got = vec![0.0f32; seq * d_model];
    let mut v_got = vec![0.0f32; seq * d_model];
    let mut e = vec![0.0f32; fetch_dim];
    for p in 0..seq {
        for t in 0..dims.num_tables {
            let dst = &mut e[t * dims.sub_dim..(t + 1) * dims.sub_dim];
            if ngram_valid(p, t, &orders, heads) {
                let src = (p * dims.num_tables + t) * dims.sub_dim;
                dst.copy_from_slice(&rows[src..src + dims.sub_dim]);
            } else {
                dst.fill(0.0);
            }
        }
        dense(
            &e,
            &key_w,
            fetch_dim,
            d_model,
            &mut k_got[p * d_model..(p + 1) * d_model],
        );
        dense(
            &e,
            &val_w,
            fetch_dim,
            d_model,
            &mut v_got[p * d_model..(p + 1) * d_model],
        );
    }
    value_conv(&mut v_got, &taps, seq, &dims);

    for (label, got, want) in [
        ("engram k", &k_got, f.get("engram_k")),
        ("engram v", &v_got, f.get("engram_v")),
    ] {
        let mut worst = 0.0f32;
        let mut sq = 0.0f64;
        for (&a, &b) in got.iter().zip(&want) {
            sq += (b as f64) * (b as f64);
            worst = worst.max((a - b).abs());
        }
        let rms = (sq / want.len() as f64).sqrt() as f32;
        let rel = worst / rms;
        println!("{label}: max abs {worst:.3e} vs RMS {rms:.3} = {rel:.3e} relative");
        assert!(rel < 1e-4, "{label} deviates by {rel:.3e} of RMS");
    }
}

#[test]
fn the_streaming_pool_matches_the_two_pass_one() {
    // The streaming form exists because materialising every cell costs
    // seq * cells * d_model floats — 504 MB at full context. It must agree
    // with the two-pass form, which is the one verified against the reference.
    let (l1, k, q, d, seq) = (5usize, 3usize, 2usize, 16usize, 40usize);
    let f = |i: usize, m: f32| ((i as f32) * m).sin();

    let head = ProbeHead {
        probes: (0..l1 * k * d).map(|i| f(i, 0.31)).collect(),
        gain: (0..l1 * k).map(|i| 1.0 + f(i, 0.17)).collect(),
        query: (0..q * d).map(|i| f(i, 0.23)).collect(),
        row_bias: (0..q * l1 * k).map(|i| f(i, 0.11) * 0.5).collect(),
        proj: (0..q * d).map(|i| f(i, 0.07)).collect(),
        bias: vec![0.25],
        cells: l1,
        probes_per_cell: k,
        queries: q,
        out_dim: 1,
    };

    // Deliberately wide dynamic range, so a naive running softmax would
    // overflow or lose the early positions entirely.
    let cells: Vec<f32> = (0..seq * l1 * d)
        .map(|i| f(i, 0.013) * (1.0 + (i % 7) as f32) * 30.0)
        .collect();

    let want = head.forward(&cells, seq, d);

    let mut pool = ProbePool::new(&head, d);
    for t in 0..seq {
        for l in 0..l1 {
            pool.observe(l, &cells[(t * l1 + l) * d..(t * l1 + l + 1) * d]);
        }
    }
    let got = pool.finish();

    assert_eq!(got.len(), want.len());
    let diff = (got[0] - want[0]).abs();
    println!(
        "probe pool: streaming {:.6} vs two-pass {:.6}, diff {diff:.3e}",
        got[0], want[0]
    );
    assert!(
        diff < 1e-4,
        "streaming pool diverged from the two-pass form: {got:?} vs {want:?}"
    );
}
