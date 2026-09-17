//! Needle 3 container parity against upstream's own reader.
//!
//! Needs:
//!   weights/needle3.cact        — `huggingface.co/Cactus-Compute/needle3`
//!   tests/cact_v3_vectors.json  — `tools/gen_cact_v3_parity.py`
//!
//! The directory is nameless and positional, so a correct tensor in the wrong
//! slot passes every per-tensor comparison. Agreeing with `export.read_export`
//! on the geometry and on all 581 records is what makes the canon checkable.

use needle_infer::cact::{CactV3, DT_CQ, DT_FP16, DT_FP32, DT_RAW};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/cact_v3_vectors.json"
);

fn fixtures() -> Option<(CactV3, serde_json::Value)> {
    if !std::path::Path::new(CACT).exists() || !std::path::Path::new(VECTORS).exists() {
        println!(
            "skipping v3 cact parity: need weights/needle3.cact and \
             tests/cact_v3_vectors.json\n  \
             hf download Cactus-Compute/needle3 needle3.cact --local-dir weights/\n  \
             PYTHONPATH=needle .venv-parity/bin/python tools/gen_cact_v3_parity.py"
        );
        return None;
    }
    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).unwrap()).unwrap();
    Some((CactV3::load(CACT).expect("v3 container should load"), want))
}

#[test]
fn geometry_matches_upstream_reader() {
    let Some((c, want)) = fixtures() else { return };
    let g = &want["geometry"];
    let u = |k: &str| g[k].as_u64().unwrap() as usize;

    assert_eq!(c.geom.num_tensors, u("num_tensors"), "num_tensors");
    assert_eq!(c.geom.vocab_size, u("vocab_size"), "vocab_size");
    assert_eq!(c.geom.out_vocab, u("out_vocab"), "out_vocab");
    assert_eq!(c.geom.d_model, u("d_model"), "d_model");
    assert_eq!(c.geom.num_heads, u("num_heads"), "num_heads");
    assert_eq!(c.geom.num_kv_heads, u("num_kv_heads"), "num_kv_heads");
    assert_eq!(c.geom.num_layers, u("num_layers"), "num_layers");
    assert_eq!(c.geom.qk_head_dim, u("qk_head_dim"), "qk_head_dim");
    assert_eq!(c.geom.v_head_dim, u("v_head_dim"), "v_head_dim");
    assert_eq!(c.geom.max_seq_len, u("max_seq_len"), "max_seq_len");
    assert_eq!(c.geom.hada_n, u("hada_n"), "hada_n");
    assert_eq!(c.geom.mhc_lanes, u("mhc_lanes"), "mhc_lanes");
    assert_eq!(c.geom.sliding_window, u("sliding_window"), "sliding_window");
    assert_eq!(c.geom.qkv_conv_taps, u("qkv_conv_taps"), "qkv_conv_taps");
    assert_eq!(c.geom.engram_slots, u("engram_slots"), "engram_slots");
    assert_eq!(c.geom.engram_sub_dim, u("engram_sub_dim"), "engram_sub_dim");
    assert_eq!(
        c.geom.num_engram_tables,
        u("num_engram_tables"),
        "num_engram_tables"
    );
    assert_eq!(
        c.geom.engram_conv_taps,
        u("engram_conv_taps"),
        "engram_conv_taps"
    );
    assert_eq!(
        c.geom.engram_conv_dilation,
        u("engram_conv_dilation"),
        "engram_conv_dilation"
    );
    assert_eq!(
        c.geom.engram_seed_heads,
        u("engram_seed_heads"),
        "engram_seed_heads"
    );
    assert_eq!(c.geom.kv_window, u("kv_window"), "kv_window");
    assert_eq!(c.geom.kv_bits, g["kv_bits"].as_u64().unwrap() as u32);
    assert_eq!(
        c.geom.rope_theta,
        g["rope_theta"].as_f64().unwrap() as f32,
        "rope_theta"
    );

    let list = |k: &str| -> Vec<usize> {
        g[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    assert_eq!(c.geom.engram_orders, list("engram_orders"), "engram_orders");
    // needle-rs calls them sites; upstream's reader calls them engram_layers.
    assert_eq!(c.geom.engram_sites, list("engram_layers"), "engram_sites");
    assert_eq!(
        c.geom.global_layers(),
        list("global_layers"),
        "global_layers"
    );

    assert_eq!(
        c.byte_len(),
        want["container_bytes"].as_u64().unwrap() as usize,
        "container size"
    );

    println!(
        "v3 geometry matches upstream: {} layers x {}, {} heads / {} kv, \
         qk {} / v {}, {} tensors, global at {:?}",
        c.geom.num_layers,
        c.geom.d_model,
        c.geom.num_heads,
        c.geom.num_kv_heads,
        c.geom.qk_head_dim,
        c.geom.v_head_dim,
        c.geom.num_tensors,
        c.geom.global_layers(),
    );
}

#[test]
fn every_directory_record_matches() {
    let Some((c, want)) = fixtures() else { return };
    let recs = want["records"].as_array().unwrap();
    assert_eq!(c.num_tensors(), recs.len(), "record count");

    for (i, w) in recs.iter().enumerate() {
        let r = c.record(i);
        assert_eq!(r.dtype, w["dtype"].as_u64().unwrap() as u8, "dtype at {i}");
        assert_eq!(r.ndim, w["ndim"].as_u64().unwrap() as u8, "ndim at {i}");
        let shape: Vec<usize> = w["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(&r.shape[..r.ndim as usize], &shape[..], "shape at {i}");
        assert_eq!(r.offset, w["offset"].as_u64().unwrap(), "offset at {i}");
        assert_eq!(r.nbytes, w["nbytes"].as_u64().unwrap(), "nbytes at {i}");
        assert_eq!(
            r.group,
            w["group"].as_u64().unwrap() as usize,
            "group at {i}"
        );
        assert_eq!(r.bits, w["bits"].as_u64().unwrap() as u8, "bits at {i}");
    }
    println!("all {} directory records match upstream", recs.len());
}

#[test]
fn quantised_tensors_decode() {
    let Some((c, _)) = fixtures() else { return };
    // Record 0 is the embedding at 4 bits; record 2 is a 2-bit projection.
    // Decoding proves the codebook survived the new header offset.
    let emb = c.cq(0).expect("embedding should decode");
    assert_eq!(emb.out_feat, c.geom.vocab_size);
    assert_eq!(emb.in_feat, c.geom.d_model);

    let q = c.cq(2).expect("q_proj should decode");
    assert_eq!(
        q.out_feat,
        c.geom.q_dim(),
        "q_proj rows == heads * qk_head_dim"
    );
    assert_eq!(q.in_feat, c.geom.d_model);

    println!(
        "cq decode ok: embedding {}x{} @4bit, q_proj {}x{} @2bit",
        emb.out_feat, emb.in_feat, q.out_feat, q.in_feat
    );
}

#[test]
fn the_container_carries_exactly_one_tokenizer_blob() {
    let Some((c, _)) = fixtures() else { return };
    let raws = c.records().iter().filter(|r| r.dtype == DT_RAW).count();
    assert_eq!(raws, 1, "exactly one RAW record carries the tokenizer");

    let blob = c.tokenizer_blob().expect("v3 container embeds a tokenizer");
    assert!(blob.len() > 1024, "tokenizer blob looks too small");

    // The mix is worth pinning: a silent change here means the quantisation
    // scheme moved and the forward pass would need revisiting.
    let n = |d: u8| c.records().iter().filter(|r| r.dtype == d).count();
    let (cq, fp16, fp32) = (n(DT_CQ), n(DT_FP16), n(DT_FP32));
    assert_eq!(
        cq + fp16 + fp32 + raws,
        c.num_tensors(),
        "every record accounted for"
    );
    // v3 introduces FP32 records, which the v2 container never carried.
    assert_eq!(fp32, 2, "v3 carries two FP32 vectors");
    println!(
        "tokenizer blob {} bytes; {cq} CQ + {fp16} FP16 + {fp32} FP32 + {raws} RAW = {}",
        blob.len(),
        c.num_tensors()
    );
}
