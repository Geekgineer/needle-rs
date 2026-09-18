//! The v3 canon, checked without the checkpoint.
//!
//! Every other v3 container test needs `weights/needle3.cact` and skips without
//! it, so a CI job that does not download 35 MB currently proves nothing about
//! `V3Layout::derive` — the walk that decides which nameless directory slot is
//! `q_proj` for layer 7. `cact.rs` already builds a synthetic **v2** container
//! for the same reason; this is the v3 one.
//!
//! The container here is assembled byte for byte from the layout in
//! `needle/model/export.py`, at a toy geometry: two layers, `d_model` 8, one
//! Engram site. Tensors are FP16 except the two Hadamard permutations, which
//! are FP32 as the exporter writes them, and the RAW tokenizer. That is enough
//! for the header parser, the canon walk and the shape validation. It says
//! nothing about numerics, which `v3_forward_parity.rs` covers against the real
//! weights.

use needle_infer::cact::{CactV3, Record, DT_FP32, DT_RAW};
use needle_infer::v3::{config_from_geometry, V3Layout, V3LayoutError};

const D: usize = 8;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const QK: usize = 4;
const VH: usize = 2;
const LANES: usize = 2;
const HADA_N: usize = 8;
const VOCAB: usize = 6;
const SLOTS: usize = 16;
const TABLES: usize = 2;
const SUB_DIM: usize = 4;
const ENGRAM_TAPS: usize = 4;
const SITE: usize = 1;
const HEAD_CONFIDENCE: f32 = 2.0;

/// One directory record plus its payload.
struct T {
    shape: Vec<usize>,
    dtype: u8,
    blob: Vec<u8>,
}

fn fp16(shape: &[usize]) -> T {
    let n: usize = shape.iter().product::<usize>().max(1);
    T {
        shape: shape.to_vec(),
        dtype: 1,
        // 1.0 in FP16 everywhere: the canon reads shapes, not values.
        blob: std::iter::repeat_n([0x00u8, 0x3C], n).flatten().collect(),
    }
}

/// FP16 with chosen values, for `heads.manifest`.
fn fp16_values(shape: &[usize], values: &[f32]) -> T {
    T {
        shape: shape.to_vec(),
        dtype: 1,
        blob: values
            .iter()
            .flat_map(|v| half_bits(*v).to_le_bytes())
            .collect(),
    }
}

/// f32 to IEEE half, adequate for the small whole numbers used here.
fn half_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3FF) as u16;
    sign | ((exp as u16) << 10) | mant
}

fn fp32_indices(n: usize) -> T {
    T {
        shape: vec![n],
        dtype: DT_FP32,
        blob: (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect(),
    }
}

fn raw(bytes: &[u8]) -> T {
    T {
        shape: Vec::new(),
        dtype: DT_RAW,
        blob: bytes.to_vec(),
    }
}

/// The 27 tensors one layer contributes when the model carries conv taps, or
/// the 24 it contributes otherwise.
fn layer_tensors(out: &mut Vec<T>, taps: usize) {
    let (ba, bb) = (2usize, 4usize); // hada_n 8 splits 2 x 4
    out.push(fp16(&[D])); // norm_in
    out.push(fp16(&[HEADS * QK, D])); // q_proj
    out.push(fp16(&[KV_HEADS * QK, D])); // k_proj
    out.push(fp16(&[KV_HEADS * VH, D])); // v_proj
    if taps > 0 {
        out.push(fp16(&[taps, HEADS * QK])); // q_taps
        out.push(fp16(&[taps, KV_HEADS * QK])); // k_taps
        out.push(fp16(&[taps, KV_HEADS * VH])); // v_taps
    }
    out.push(fp16(&[QK])); // q_norm
    out.push(fp16(&[QK])); // k_norm
    out.push(fp16(&[HEADS * VH, D])); // gate_proj
    out.push(fp16(&[D, HEADS * VH])); // out_proj
    out.push(fp16(&[D])); // post_norm
    out.push(fp16(&[1])); // attn_gate
    out.push(fp16(&[D])); // pre_hada
    for _ in 0..5 {
        out.push(fp16(&[HADA_N])); // d1, d2, b2, d3, d4
    }
    for _ in 0..3 {
        out.push(fp16(&[ba, ba])); // w{1,2,3}a
        out.push(fp16(&[bb, bb])); // w{1,2,3}b
    }
    out.push(fp16(&[D, 2])); // cond_v, rank 2
    out.push(fp16(&[2, HADA_N])); // cond_u
}

struct Options {
    taps: usize,
    heads: bool,
    tokenizer: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            taps: 3,
            heads: true,
            tokenizer: true,
        }
    }
}

fn tensors(o: &Options) -> Vec<T> {
    let mut ts = vec![fp16(&[VOCAB, D])]; // embedding
    for _ in 0..LAYERS {
        layer_tensors(&mut ts, o.taps);
    }
    ts.push(fp16(&[LAYERS])); // mhc a_pre
    ts.push(fp16(&[LAYERS])); // a_post
    ts.push(fp16(&[LAYERS])); // a_res
    ts.push(fp16(&[LAYERS, LANES])); // b_pre
    ts.push(fp16(&[LAYERS, LANES])); // b_post
    ts.push(fp16(&[LAYERS, LANES, LANES])); // b_res
    ts.push(fp16(&[LAYERS * LANES, LANES * D])); // phi_pre
    ts.push(fp16(&[LAYERS * LANES, LANES * D])); // phi_post
    ts.push(fp16(&[LAYERS * LANES * LANES, LANES * D])); // phi_res
    ts.push(fp32_indices(HADA_N)); // hada_p1
    ts.push(fp32_indices(HADA_N)); // hada_p2
    ts.push(fp16(&[TABLES * SLOTS, SUB_DIM])); // engram tables
    ts.push(fp16(&[D, TABLES * SUB_DIM])); // engram key_proj
    ts.push(fp16(&[D, TABLES * SUB_DIM])); // engram value_proj
    ts.push(fp16(&[ENGRAM_TAPS, D])); // engram taps
    ts.push(fp16(&[D])); // final_norm
    if o.heads {
        let rows = LAYERS + 1;
        let (k, q) = (2usize, 2usize);
        ts.push(fp16_values(&[1], &[HEAD_CONFIDENCE])); // heads.manifest
        ts.push(fp16(&[rows * k, D])); // probes
        ts.push(fp16(&[rows, k])); // gain
        ts.push(fp16(&[q, D])); // query
        ts.push(fp16(&[q, rows, k])); // row_bias
        ts.push(fp16(&[1, q * D])); // proj
        ts.push(fp16(&[1])); // bias
    }
    if o.tokenizer {
        ts.push(raw(b"a tokenizer blob lives here"));
    }
    ts
}

fn build(o: &Options) -> Vec<u8> {
    const HEADER: usize = 49 * 4;
    const REC: usize = 44;
    let ts = tensors(o);
    let cb_len = needle_core::cq::CODEBOOK_LEN;

    // Layer 1 attends globally, layer 0 locally.
    let gmask: u64 = 1 << 1;

    // Kept as a table: this is the header layout from export.py, field for
    // field, and reading it against that docstring is the point.
    #[rustfmt::skip]
    let fields: [u32; 48] = [
        0x05E1_2A84,        // tag
        ts.len() as u32,    // num_tensors
        cb_len as u32,      // codebook_len
        0,                  // kv_window
        8,                  // kv_bits
        VOCAB as u32,       // vocab
        VOCAB as u32,       // out_vocab
        D as u32,
        HEADS as u32,
        KV_HEADS as u32,
        LAYERS as u32,
        QK as u32,
        VH as u32,
        32,                 // max_seq_len
        HADA_N as u32,
        LANES as u32,
        4,                  // sliding_window
        (gmask & 0xFFFF_FFFF) as u32,
        (gmask >> 32) as u32,
        o.taps as u32,
        SLOTS as u32,
        SUB_DIM as u32,
        TABLES as u32,
        ENGRAM_TAPS as u32,
        3,                  // engram_conv_dilation
        0,                  // engram_seed_heads
        2,                  // num_engram_orders
        2, 3, 0, 0,         // orders[4]
        1,                  // num_engram_sites
        SITE as u32, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, // sites[16] tail
    ];
    let mut out = Vec::new();
    for f in fields {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out.extend_from_slice(&100_000.0f32.to_le_bytes());
    assert_eq!(out.len(), HEADER);
    for _ in 0..cb_len {
        out.extend_from_slice(&0.5f32.to_le_bytes());
    }

    // Blobs are 64-byte aligned and follow the directory.
    let mut pos = out.len() + ts.len() * REC;
    let mut offsets = Vec::with_capacity(ts.len());
    for t in &ts {
        pos = pos.div_ceil(64) * 64;
        offsets.push(pos);
        pos += t.blob.len();
    }
    for (i, t) in ts.iter().enumerate() {
        out.push(t.dtype);
        out.push(t.shape.len() as u8);
        out.extend_from_slice(&0u16.to_le_bytes());
        for k in 0..4 {
            out.extend_from_slice(&(t.shape.get(k).copied().unwrap_or(0) as u32).to_le_bytes());
        }
        out.extend_from_slice(&(offsets[i] as u64).to_le_bytes());
        out.extend_from_slice(&(t.blob.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // group
        out.extend_from_slice(&0u32.to_le_bytes()); // bits
    }
    for (i, t) in ts.iter().enumerate() {
        out.resize(offsets[i], 0);
        out.extend_from_slice(&t.blob);
    }
    out
}

fn shape_offset(index: usize, axis: usize) -> usize {
    49 * 4 + needle_core::cq::CODEBOOK_LEN * 4 + index * 44 + 4 + axis * 4
}

fn layout_of(bytes: Vec<u8>) -> Result<(V3Layout, Vec<Record>), V3LayoutError> {
    let c = CactV3::from_bytes(bytes).expect("header should parse");
    let cfg = config_from_geometry(&c.geom).expect("geometry should be consistent");
    V3Layout::derive(&cfg, c.records()).map(|l| (l, c.records().to_vec()))
}

#[test]
fn the_synthetic_container_parses_as_declared() {
    let c = CactV3::from_bytes(build(&Options::default())).expect("parse");
    let g = &c.geom;
    assert_eq!(g.d_model, D);
    assert_eq!(g.num_layers, LAYERS);
    assert_eq!(g.qk_head_dim, QK);
    assert_eq!(g.v_head_dim, VH);
    assert_eq!(g.qkv_conv_taps, 3);
    assert_eq!(g.sliding_window, 4);
    assert_eq!(g.global_layers(), vec![1]);
    assert!(!g.is_global(0));
    assert_eq!(g.engram_sites, vec![SITE]);
    assert_eq!(g.engram_orders, vec![2, 3]);
    assert_eq!(g.rope_theta, 100_000.0);

    let cfg = config_from_geometry(g).expect("geometry");
    assert_eq!(cfg.q_dim(), HEADS * QK);
    assert_eq!(cfg.k_dim(), KV_HEADS * QK);
    assert_eq!(cfg.v_dim(), KV_HEADS * VH);
    assert_eq!(cfg.attn_out_dim(), HEADS * VH);
    assert_eq!(cfg.engram.num_tables(), TABLES);
}

/// The anchors the walk has to land on, computed by hand from the canon:
/// 1 embedding, 27 per layer, 9 mHC, 2 permutations, 4 per site, final norm,
/// then the head block and the tokenizer.
#[test]
fn the_canon_walk_lands_on_the_right_slots() {
    let (l, _) = layout_of(build(&Options::default())).expect("canon should walk");
    assert_eq!(l.embedding, 0);
    assert_eq!(l.layers.len(), LAYERS);
    assert_eq!(l.layers[0].norm_in, 1);
    assert_eq!(l.layers[0].qkv_taps, Some([5, 6, 7]));
    assert_eq!(l.layers[0].q_norm, 8);
    assert_eq!(l.layers[0].mlp[0], 15);
    assert_eq!(l.layers[1].norm_in, 28);
    assert_eq!(l.mhc.scalars, [55, 56, 57, 58, 59, 60]);
    assert_eq!(l.mhc.phi, [61, 62, 63]);
    assert_eq!(l.hada_perms, [64, 65]);
    assert_eq!(l.engrams.len(), 1);
    assert_eq!(l.engrams[0].tables, 66);
    assert_eq!(l.engrams[0].taps, 69);
    assert_eq!(l.final_norm, 70);
    assert_eq!(l.head_manifest, Some(71));
    assert_eq!(l.head_tensors, (72..78).collect::<Vec<_>>());
    assert_eq!(l.tokenizer, Some(78));
}

/// The permutations are the only FP32 records, which is how a reader tells them
/// apart from the FP16 diagonals around them.
#[test]
fn the_hadamard_permutations_are_the_fp32_records() {
    let (l, records) = layout_of(build(&Options::default())).expect("canon");
    let fp32: Vec<usize> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.dtype == DT_FP32)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(fp32, l.hada_perms.to_vec());
    for idx in l.hada_perms {
        assert_eq!(records[idx].shape[0], HADA_N);
    }
}

/// With `qkv_conv_taps == 0` a layer contributes 24 tensors, not 27, and every
/// slot after the first layer moves by three per layer. Getting this wrong
/// would read the taps of one model as the norms of another.
#[test]
fn a_container_without_conv_taps_shifts_the_whole_canon() {
    let o = Options {
        taps: 0,
        ..Default::default()
    };
    let (l, _) = layout_of(build(&o)).expect("canon should walk");
    assert_eq!(l.layers[0].qkv_taps, None);
    assert_eq!(l.layers[0].q_norm, 5);
    assert_eq!(l.layers[1].norm_in, 25);
    assert_eq!(l.mhc.scalars[0], 49);
    assert_eq!(l.final_norm, 64);
}

#[test]
fn a_container_without_heads_or_tokenizer_still_resolves() {
    let o = Options {
        heads: false,
        tokenizer: false,
        ..Default::default()
    };
    let (l, records) = layout_of(build(&o)).expect("canon should walk");
    assert_eq!(l.head_manifest, None);
    assert!(l.head_tensors.is_empty());
    assert_eq!(l.tokenizer, None);
    assert_eq!(l.final_norm, records.len() - 1);
}

/// A tensor in the wrong slot is the failure the shape check exists for: the
/// directory has no names, so nothing else would notice.
#[test]
fn a_wrong_shape_in_a_canon_slot_is_rejected() {
    let bytes = build(&Options::default());
    let (l, _) = layout_of(bytes.clone()).expect("canon");
    let mut broken = bytes;
    let off = shape_offset(l.layers[0].q_proj, 0);
    broken[off..off + 4].copy_from_slice(&7u32.to_le_bytes());

    match layout_of(broken) {
        Err(V3LayoutError::BadShape { index, what, .. }) => {
            assert_eq!(index, l.layers[0].q_proj);
            assert_eq!(what, "q_proj");
        }
        Err(other) => panic!("expected a shape mismatch, got {other}"),
        Ok(_) => panic!("a wrong shape in a canon slot must not resolve"),
    }
}

/// The same check has to hold deep in the canon, where a drift of one layer
/// puts an Engram table where a projection belongs.
#[test]
fn a_wrong_engram_shape_is_rejected() {
    let bytes = build(&Options::default());
    let (l, _) = layout_of(bytes.clone()).expect("canon");
    let mut broken = bytes;
    let off = shape_offset(l.engrams[0].tables, 1);
    broken[off..off + 4].copy_from_slice(&9u32.to_le_bytes());

    match layout_of(broken) {
        Err(V3LayoutError::BadShape { what, .. }) => assert_eq!(what, "engram tables"),
        Err(other) => panic!("expected a shape mismatch, got {other}"),
        Ok(_) => panic!("a wrong engram shape must not resolve"),
    }
}

/// A directory that ends before the canon does must say so, rather than
/// indexing past its own records.
#[test]
fn a_short_directory_is_rejected() {
    let bytes = build(&Options::default());
    let mut broken = bytes;
    // num_tensors is header word 1; claim the file holds only the embedding
    // and the first layer.
    broken[4..8].copy_from_slice(&20u32.to_le_bytes());
    match layout_of(broken) {
        Err(V3LayoutError::TooFewTensors { need, got }) => {
            assert!(need > got, "need {need} should exceed got {got}");
            assert_eq!(got, 20);
        }
        Err(other) => panic!("expected a short directory, got {other}"),
        Ok(_) => panic!("a short directory must not resolve"),
    }
}
