//! Needle 3 whole-model parity.
//!
//! Needs:
//!   weights/needle3.cact           — `huggingface.co/Cactus-Compute/needle3`
//!   tests/v3_forward_vectors.json  — `tools/gen_v3_forward_parity.py`
//!   tests/v3_forward_vectors.f32
//!
//! The component tests prove each kernel in isolation. This proves they are
//! wired together in the right order, with the right weights in the right
//! slots, over a real prompt — which no amount of component parity can.

use needle_core::v3::V3Cache;
use needle_infer::cact::CactV3;
use needle_infer::v3::model_from_cact;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_forward_vectors.json"
);
const F32: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_forward_vectors.f32"
);

struct Fixture {
    meta: serde_json::Value,
    floats: Vec<f32>,
}

impl Fixture {
    fn load() -> Option<Self> {
        for p in [CACT, JSON, F32] {
            if !std::path::Path::new(p).exists() {
                println!(
                    "skipping v3 forward parity: missing {p}\n  \
                     JAX_PLATFORMS=cpu PYTHONPATH=needle:tools \
                     .venv-parity/bin/python tools/gen_v3_forward_parity.py"
                );
                return None;
            }
        }
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(JSON).unwrap()).unwrap();
        let floats = std::fs::read(F32)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self { meta, floats })
    }

    fn stage(&self, name: &str) -> Vec<f32> {
        let e = &self.meta["stages"][name];
        let off = e["offset"].as_u64().unwrap() as usize;
        let len = e["len"].as_u64().unwrap() as usize;
        self.floats[off..off + len].to_vec()
    }
}

#[test]
fn logits_match_the_reference() {
    let Some(f) = Fixture::load() else { return };

    let tokens: Vec<u32> = f.meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let rows = f.meta["geometry"]["out_vocab"].as_u64().unwrap() as usize;

    let cact = CactV3::load(CACT).expect("load container");
    let model = model_from_cact(&cact).expect("build model");

    let t0 = std::time::Instant::now();
    let got = model.forward_sequence(&tokens);
    let elapsed = t0.elapsed();

    let want = f.stage("logits");
    assert_eq!(got.len(), want.len(), "logit count");
    assert_eq!(got.len(), tokens.len() * rows);

    let mut worst = 0.0f32;
    let mut sq = 0.0f64;
    for (&a, &b) in got.iter().zip(&want) {
        sq += (b as f64) * (b as f64);
        worst = worst.max((a - b).abs());
    }
    let rms = (sq / want.len() as f64).sqrt() as f32;
    let rel = worst / rms;

    // The decisive check: the same greedy token at every position. Float
    // deviation matters only insofar as it changes an argmax.
    let want_argmax: Vec<usize> = f.meta["argmax"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut mismatches = Vec::new();
    for (t, expect) in want_argmax.iter().enumerate() {
        let row = &got[t * rows..(t + 1) * rows];
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if best != *expect {
            mismatches.push((t, best, *expect));
        }
    }

    println!(
        "v3 forward: {} positions, max abs deviation {worst:.3e} against logit \
         RMS {rms:.3} = {rel:.3e} relative, {} argmax mismatches, {:?}",
        tokens.len(),
        mismatches.len(),
        elapsed
    );
    assert!(
        mismatches.is_empty(),
        "greedy token differs at {:?} (position, got, want)",
        &mismatches[..mismatches.len().min(8)]
    );
    assert!(
        rel < 5e-3,
        "logits deviate by {rel:.3e} of RMS (abs {worst:.3e})"
    );
}

#[test]
fn incremental_decode_reproduces_the_prefill() {
    let Some(f) = Fixture::load() else { return };
    let tokens: Vec<u32> = f.meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let rows = f.meta["geometry"]["out_vocab"].as_u64().unwrap() as usize;

    let cact = CactV3::load(CACT).expect("load container");
    let model = model_from_cact(&cact).expect("build model");

    // The prefill path is the one verified against the reference, so it is
    // the target here: the cache must reproduce it, not merely look sane.
    let want = model.forward_sequence(&tokens);

    let mut cache = V3Cache::new(&model.cfg, tokens.len());
    let t0 = std::time::Instant::now();
    let mut worst = 0.0f32;
    let mut sq = 0.0f64;
    let mut mismatches = Vec::new();

    for (t, &tok) in tokens.iter().enumerate() {
        let got = model.decode_step(&mut cache, tok);
        assert_eq!(got.len(), rows);
        let expect = &want[t * rows..(t + 1) * rows];

        for (&a, &b) in got.iter().zip(expect) {
            sq += (b as f64) * (b as f64);
            worst = worst.max((a - b).abs());
        }
        let am = |r: &[f32]| {
            r.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        if am(&got) != am(expect) {
            mismatches.push(t);
        }
    }
    let elapsed = t0.elapsed();
    let rms = (sq / want.len() as f64).sqrt() as f32;
    let rel = worst / rms;

    println!(
        "v3 decode: {} steps, max abs deviation {worst:.3e} against RMS {rms:.3} \
         = {rel:.3e} relative, {} argmax mismatches, {:?} ({:.1} ms/token), \
         cache {} KB",
        tokens.len(),
        mismatches.len(),
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / tokens.len() as f64,
        cache.bytes() / 1024,
    );
    assert!(
        mismatches.is_empty(),
        "incremental decode diverged from prefill at positions {mismatches:?}"
    );
    assert!(
        rel < 1e-4,
        "decode deviates from prefill by {rel:.3e} of RMS (abs {worst:.3e})"
    );
}
