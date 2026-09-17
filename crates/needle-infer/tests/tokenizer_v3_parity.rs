//! Needle 3 tokenizer parity.
//!
//! Needs:
//!   weights/needle3.cact             — `huggingface.co/Cactus-Compute/needle3`
//!   tests/tokenizer_v3_vectors.json  — `tools/gen_tokenizer_v3_parity.py`
//!
//! v3 embeds its own SentencePiece model and it is **not** v2's — the two
//! `tokenizer.model` files differ (126,520 vs 132,396 bytes). Reusing the v2
//! vocabulary for v3 would tokenize plausibly and decode to nonsense, so this
//! checks the container's own blob against real `sentencepiece` output.
//!
//! Upstream's tokenizer header format did not change between generations, so
//! the existing reader serves v3 unaltered. This test is what makes that a
//! fact rather than an assumption.

use needle_infer::cact::CactV3;
use needle_infer::sp_tokenizer::SpTokenizer;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/tokenizer_v3_vectors.json"
);

fn fixtures() -> Option<(SpTokenizer, serde_json::Value)> {
    if !std::path::Path::new(CACT).exists() || !std::path::Path::new(VECTORS).exists() {
        println!(
            "skipping v3 tokenizer parity: need weights/needle3.cact and \
             tests/tokenizer_v3_vectors.json\n  \
             PYTHONPATH=needle .venv-parity/bin/python tools/gen_tokenizer_v3_parity.py"
        );
        return None;
    }
    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).unwrap()).unwrap();
    let cact = CactV3::load(CACT).expect("load v3 container");
    let blob = cact
        .tokenizer_blob()
        .expect("v3 container embeds a tokenizer");
    Some((
        SpTokenizer::from_blob(blob).expect("v3 tokenizer blob should decode"),
        want,
    ))
}

#[test]
fn vocab_matches_the_container() {
    let Some((t, want)) = fixtures() else { return };
    assert_eq!(
        t.vocab_size(),
        want["pieces"].as_u64().unwrap() as usize,
        "piece count"
    );
    println!("v3 tokenizer: {} pieces", t.vocab_size());
}

#[test]
fn encode_matches_sentencepiece_exactly() {
    let Some((t, want)) = fixtures() else { return };
    let mut n = 0usize;
    for case in want["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expect: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(t.encode(text), expect, "encode {text:?}");
        n += 1;
    }
    println!("v3 tokenizer: {n} cases match sentencepiece exactly");
}

#[test]
fn round_trips_through_decode() {
    let Some((t, want)) = fixtures() else { return };
    for case in want["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        if text.is_empty() {
            continue;
        }
        let ids = t.encode(text);
        let back = t.decode(&ids);
        assert_eq!(back.trim(), text.trim(), "round trip {text:?}");
    }
    println!("v3 tokenizer: every non-empty case round-trips");
}

#[test]
fn carries_the_v3_special_tokens() {
    let Some((t, _)) = fixtures() else { return };
    // v3 extends the special-token block beyond v2's chat/tool markers with
    // reasoning, extraction and modality tokens. They must resolve to ids, or
    // prompt construction would silently emit them as literal text.
    for surface in [
        "<|im_start|>",
        "<|im_end|>",
        "<tools>",
        "</tools>",
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
        "<extract>",
        "<schema>",
    ] {
        let id = t
            .id_of(surface)
            .unwrap_or_else(|| panic!("{surface} should be a single token"));
        assert_eq!(t.piece(id), Some(surface), "piece for {surface}");
    }
    println!("v3 special tokens resolve, including <think> and <extract>");
}
