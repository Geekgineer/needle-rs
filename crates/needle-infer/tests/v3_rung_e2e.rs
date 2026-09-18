//! A real ladder rung, end to end.
//!
//! `v3_rung_containers` proves the *shapes* resolve using synthetic files. This
//! runs actual rung containers built from upstream's checkpoint by
//! `tools/build_rung.py`, which calls upstream's own `rung()` —
//! `ladder_slice` + `ladder_config` — and its exporter. Nothing here is
//! reimplemented.
//!
//! The point is that no engine change was needed: a rung is an ordinary v3
//! container with a smaller `num_layers` and remapped global/Engram layers, and
//! the geometry-driven loader reads it as it stands.
//!
//! Build the inputs (needs JAX and the 242 MB base checkpoint — see
//! docs/v3-port-record.md):
//!
//! ```text
//! python3 tools/build_rung.py <needle3.safetensors> weights/needle3.cact weights/ 6,8
//! ```
//!
//! Skips cleanly when they are absent, like every other weight-backed suite.
//!
//! Run: cargo test -p needle-infer --release --test v3_rung_e2e -- --nocapture

use needle_infer::v3_engine::{extract_tool_call, V3Engine, V3Options};

const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;

fn rung(depth: usize) -> Option<(V3Engine, usize)> {
    let p = format!(
        "{}/../../weights/needle3-{depth}l.cact",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&p).exists() {
        eprintln!("skipping {depth}L: no {p}");
        return None;
    }
    let e = V3Engine::load(&p).unwrap_or_else(|err| panic!("{depth}L failed to load: {err}"));
    Some((e, depth))
}

#[test]
fn a_rung_declares_the_depth_it_was_built_at() {
    for depth in [6usize, 8] {
        let Some((e, d)) = rung(depth) else { continue };
        assert_eq!(e.model.cfg.num_layers, d, "{d}L: header num_layers");

        // ladder_config remaps both of these onto the surviving blocks, so they
        // are the clearest signal that we are reading a sliced file rather than
        // a truncated one.
        let globals: Vec<usize> = (0..d)
            .filter(|&l| e.model.cfg.attention_span(l).is_none())
            .collect();
        let want_global: Vec<usize> = match d {
            6 => vec![1, 3, 4, 5],
            8 => vec![1, 3, 5, 7],
            _ => unreachable!(),
        };
        let want_sites: Vec<usize> = match d {
            6 => vec![5],
            8 => vec![4, 7],
            _ => unreachable!(),
        };
        assert_eq!(globals, want_global, "{d}L: global-attention layers");
        assert_eq!(e.model.cfg.engram.sites, want_sites, "{d}L: Engram sites");

        // Every rung keeps block 0 and the last block, and the last block
        // carries both a global layer and an Engram site upstream.
        assert!(globals.contains(&(d - 1)), "{d}L: the last block is global");
        assert!(
            e.model.cfg.engram.sites.contains(&(d - 1)),
            "{d}L: the last block holds an Engram site"
        );
        println!(
            "  {d}L: global {globals:?}, engram sites {:?}",
            e.model.cfg.engram.sites
        );
    }
}

#[test]
fn a_rung_answers_and_costs_less_cache() {
    let mut seen = 0;
    let mut prev_kv = 0usize;
    for depth in [6usize, 8] {
        let Some((e, d)) = rung(depth) else { continue };
        let res = e.generate(
            "What's the weather in Paris?",
            TOOLS,
            &V3Options {
                max_new_tokens: 96,
                ..Default::default()
            },
        );
        let call = extract_tool_call(&res.text);
        let kv = e.model.cfg.kv_bytes(512, 4);
        println!(
            "  {d}L -> {call:?}  (kv at 512 tokens: {:.1} MB)",
            kv as f64 / 1048576.0
        );
        assert_eq!(
            call.as_deref(),
            Some(r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#),
            "{d}L produced the wrong call"
        );
        assert!(kv > prev_kv, "{d}L: cache did not grow with depth");
        prev_kv = kv;
        seen += 1;
    }
    if seen == 0 {
        eprintln!("no rung containers present; nothing asserted");
    }
}

#[test]
fn a_rung_keeps_the_invariants_the_full_model_has() {
    let Some((e, d)) = rung(8) else { return };
    // The same prefill/decode agreement the 20-layer suite asserts. A rung
    // changes the geometry, not the contract.
    let prompt = V3Engine::build_prompt("What's the weather in Paris?", TOOLS, None);
    let tokens = e.tokenizer.encode(&prompt);
    let model = &e.model;

    let mut stepped = needle_core::v3::V3Cache::new(&model.cfg, tokens.len() + 8);
    let mut logits = Vec::new();
    for &t in &tokens {
        logits = model.decode_step(&mut stepped, t);
    }
    let mut filled = needle_core::v3::V3Cache::new(&model.cfg, tokens.len() + 8);
    let pre = model.prefill(&tokens, &mut filled);

    let worst = pre
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  {d}L: prefill vs stepping, max abs logit diff {worst:.3e}");
    assert_eq!(worst, 0.0, "{d}L: batched prefill diverged from stepping");
}
