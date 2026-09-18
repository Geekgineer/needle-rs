//! Per-rung forward parity: our load-time ladder slice against upstream's.
//!
//! The 20-block suite earns the token-exact claim. Depth selection deserves the
//! same standard, not a weaker one, because a rung is what a memory-constrained
//! caller will actually run.
//!
//! Both sides start from the *same* `needle3.cact`. `tools/gen_v3_ladder_parity.py`
//! rebuilds the parameter tree from that container, applies upstream's own
//! `rung()` — `ladder_slice` + `ladder_config` — and records the logits;
//! `V3Engine::load_with_depth` slices the same bytes here. Comparing against a
//! container built by `needle build --layers N` instead would confound a
//! slicing error with a requantisation difference, since the published archive
//! is 2-bit for most tensors while the public exporter emits 4.
//!
//! Measured the way the 20-block ladder is: max absolute deviation against the
//! RMS of the reference logits, plus argmax mismatches, which is what actually
//! decides a token.
//!
//! Fixtures are gitignored for size — regenerate with:
//!
//! ```text
//! JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
//!     tools/gen_v3_ladder_parity.py
//! ```
//!
//! Run: cargo test -p needle-infer --release --test v3_ladder_parity -- --nocapture

use needle_infer::v3_engine::V3Engine;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_ladder_vectors.json"
);
const F32: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_ladder_vectors.f32"
);

struct Fixture {
    meta: serde_json::Value,
    floats: Vec<f32>,
}

impl Fixture {
    fn load() -> Option<Self> {
        if !std::path::Path::new(CACT).exists() {
            eprintln!("skipping: no {CACT}");
            return None;
        }
        let raw = std::fs::read_to_string(JSON).ok().or_else(|| {
            eprintln!("skipping: no {JSON} — see the module docs to regenerate");
            None
        })?;
        let bytes = std::fs::read(F32).ok()?;
        let floats = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self {
            meta: serde_json::from_str(&raw).ok()?,
            floats,
        })
    }

    fn slice(&self, v: &serde_json::Value) -> &[f32] {
        let off = v["off"].as_u64().unwrap() as usize;
        let len = v["len"].as_u64().unwrap() as usize;
        &self.floats[off..off + len]
    }
}

#[test]
fn every_rung_matches_upstreams_own_slice() {
    let Some(f) = Fixture::load() else { return };
    let tokens: Vec<u32> = f.meta["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let mut worst_rel_overall = 0.0f32;
    let mut total_mismatches = 0usize;

    for r in f.meta["rungs"].as_array().unwrap() {
        let depth = r["depth"].as_u64().unwrap() as usize;
        let rows = r["rows"].as_u64().unwrap() as usize;
        let want = f.slice(&r["logits"]);

        let engine = V3Engine::load_with_depth(CACT, depth)
            .unwrap_or_else(|e| panic!("depth {depth}: load_with_depth failed: {e}"));

        // The geometry upstream's ladder_config produced, independently of the
        // numbers: if these disagree the logits comparison is meaningless.
        let globals: Vec<u64> = (0..depth)
            .filter(|&l| engine.model.cfg.attention_span(l).is_none())
            .map(|l| l as u64)
            .collect();
        let want_globals: Vec<u64> = r["global_layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(
            globals, want_globals,
            "depth {depth}: global-attention layers"
        );

        let got = engine.model.forward_sequence(&tokens);
        assert_eq!(got.len(), want.len(), "depth {depth}: logit count");

        let mut worst = 0.0f32;
        let mut sq = 0.0f64;
        for (a, b) in got.iter().zip(want) {
            worst = worst.max((a - b).abs());
            sq += (*b as f64) * (*b as f64);
        }
        let rms = (sq / want.len() as f64).sqrt() as f32;
        let rel = worst / rms;

        // Deviation matters insofar as it changes a token.
        let want_argmax: Vec<usize> = r["argmax"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut mismatches = 0;
        for (t, expect) in want_argmax.iter().enumerate() {
            let row = &got[t * rows..(t + 1) * rows];
            let got_arg = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if got_arg != *expect {
                mismatches += 1;
            }
        }

        println!(
            "  {depth:>2} blocks: max abs {worst:.3e} against RMS {rms:.3} = {rel:.3e} relative, \
             {mismatches} argmax mismatches over {} positions",
            want_argmax.len()
        );
        worst_rel_overall = worst_rel_overall.max(rel);
        total_mismatches += mismatches;

        assert_eq!(
            mismatches, 0,
            "depth {depth}: {mismatches} argmax mismatches — the slice picks different tokens"
        );
        assert!(
            rel < 5e-3,
            "depth {depth}: logits deviate by {rel:.3e} of RMS (abs {worst:.3e})"
        );
    }

    println!(
        "worst rung: {worst_rel_overall:.3e} relative, {total_mismatches} argmax mismatches in total"
    );
}

/// The full depth taken through the slicing path must equal the ordinary load.
///
/// `load_with_depth(n)` where `n` is the container's own depth short-circuits,
/// but that short-circuit is exactly the kind of thing that rots. Compared
/// bit-for-bit rather than approximately.
#[test]
fn asking_for_the_full_depth_is_the_ordinary_load() {
    if !std::path::Path::new(CACT).exists() {
        return;
    }
    let full = V3Engine::load(CACT).expect("load");
    let n = full.model.cfg.num_layers;
    let viadepth = V3Engine::load_with_depth(CACT, n).expect("load_with_depth");

    let tokens: Vec<u32> = vec![2, 100, 200, 300, 400];
    let a = full.model.forward_sequence(&tokens);
    let b = viadepth.model.forward_sequence(&tokens);
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("full depth via the slicing path: max abs diff {worst:.3e}");
    assert_eq!(
        worst, 0.0,
        "the full-depth path diverged from the plain load"
    );
}
