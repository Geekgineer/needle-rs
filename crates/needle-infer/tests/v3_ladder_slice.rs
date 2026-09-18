//! Slicing a ladder rung out of the full container at load time.
//!
//! `needle build --layers N` ships a rung as its own file. `load_with_depth`
//! takes the same slice from the 20-block container, so one download serves
//! every depth. This asserts the two agree.
//!
//! The comparison is structural and behavioural, not bit-exact, and that is a
//! property of the artifacts rather than a weakness of the test: the published
//! container is quantised at 2 bits for most tensors while the public exporter
//! can only emit 4 (`_cq_pack` raises otherwise), so a locally built rung is a
//! different quantisation of the same weights. Geometry must match exactly;
//! outputs are compared as tool calls.
//!
//! Rung files are optional — see `tools/build_rung.py`. Without them the slice
//! is still checked against the ladder and against itself.
//!
//! Run: cargo test -p needle-infer --release --test v3_ladder_slice -- --nocapture

use needle_infer::v3_engine::{extract_tool_call, V3Engine, V3Options};

const FULL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
const QUERY: &str = "What's the weather in Paris?";

fn have() -> bool {
    if std::path::Path::new(FULL).exists() {
        return true;
    }
    eprintln!("skipping: no {FULL}");
    false
}

fn globals(e: &V3Engine) -> Vec<usize> {
    (0..e.model.cfg.num_layers)
        .filter(|&l| e.model.cfg.attention_span(l).is_none())
        .collect()
}

#[test]
fn a_slice_has_the_geometry_the_ladder_prescribes() {
    if !have() {
        return;
    }
    // Upstream's ladder_config output, via tests/v3_ladder_rungs.json.
    for (depth, want_global, want_sites) in [
        (6usize, vec![1usize, 3, 4, 5], vec![5usize]),
        (8, vec![1, 3, 5, 7], vec![4, 7]),
        (12, vec![2, 5, 8, 11], vec![4, 6, 11]),
        (16, vec![4, 9, 12, 15], vec![3, 7, 10, 15]),
    ] {
        let e = V3Engine::load_with_depth(FULL, depth).expect("slice");
        assert_eq!(e.model.cfg.num_layers, depth);
        assert_eq!(globals(&e), want_global, "{depth}L global layers");
        assert_eq!(
            e.model.cfg.engram.sites, want_sites,
            "{depth}L Engram sites"
        );
        // The head is sliced to one cell per kept block plus the embedding.
        if let Some(h) = &e.confidence {
            assert_eq!(h.cells, depth + 1, "{depth}L head cells");
        }
        println!(
            "  {depth}L: global {:?}, sites {:?}",
            globals(&e),
            e.model.cfg.engram.sites
        );
    }
}

#[test]
fn a_slice_agrees_with_a_container_built_at_that_depth() {
    if !have() {
        return;
    }
    let mut compared = 0;
    for depth in [6usize, 8] {
        let built_path = format!(
            "{}/../../weights/needle3-{depth}l.cact",
            env!("CARGO_MANIFEST_DIR")
        );
        if !std::path::Path::new(&built_path).exists() {
            eprintln!("  {depth}L: no built rung to compare against");
            continue;
        }
        let sliced = V3Engine::load_with_depth(FULL, depth).expect("slice");
        let built = V3Engine::load(&built_path).expect("built rung");

        // The structure must be identical: same blocks, same roles.
        assert_eq!(sliced.model.cfg.num_layers, built.model.cfg.num_layers);
        assert_eq!(globals(&sliced), globals(&built), "{depth}L global layers");
        assert_eq!(
            sliced.model.cfg.engram.sites, built.model.cfg.engram.sites,
            "{depth}L Engram sites"
        );
        assert_eq!(
            sliced.model.cfg.kv_bytes(512, 4),
            built.model.cfg.kv_bytes(512, 4),
            "{depth}L cache size"
        );
        println!(
            "  {depth}L: slice and build agree on geometry (global {:?}, sites {:?})",
            globals(&sliced),
            sliced.model.cfg.engram.sites
        );
        compared += 1;
    }
    if compared == 0 {
        eprintln!("no built rungs present; geometry comparison skipped");
    }
}

#[test]
fn a_slice_answers_from_six_blocks_up() {
    if !have() {
        return;
    }
    let opts = V3Options {
        max_new_tokens: 96,
        ..Default::default()
    };
    for depth in [8usize, 12, 16, 20] {
        let e = V3Engine::load_with_depth(FULL, depth).expect("slice");
        let call = extract_tool_call(&e.generate(QUERY, TOOLS, &opts).text);
        println!("  {depth}L -> {call:?}");
        assert_eq!(
            call.as_deref(),
            Some(r#"[{"name":"get_weather","arguments":{"city":"Paris"}}]"#),
            "{depth}L did not produce the expected call"
        );
    }
}

#[test]
fn a_shallow_rung_costs_less_cache_and_the_full_depth_is_unchanged() {
    if !have() {
        return;
    }
    let full = V3Engine::load(FULL).expect("full");
    // Asking for the container's own depth must be the ordinary load.
    let same = V3Engine::load_with_depth(FULL, full.model.cfg.num_layers).expect("full via depth");
    assert_eq!(same.model.cfg.num_layers, full.model.cfg.num_layers);
    assert_eq!(globals(&same), globals(&full));
    assert_eq!(same.model.cfg.engram.sites, full.model.cfg.engram.sites);

    let mut prev = 0usize;
    for depth in [6usize, 8, 12, 16, 20] {
        let e = V3Engine::load_with_depth(FULL, depth).expect("slice");
        let kv = e.model.cfg.kv_bytes(512, 4);
        assert!(kv > prev, "{depth}L cache did not grow with depth");
        prev = kv;
        println!(
            "  {depth:>2}L: {:.1} MB of cache at 512 tokens",
            kv as f64 / 1048576.0
        );
    }
}

#[test]
fn a_depth_outside_the_ladder_is_refused() {
    if !have() {
        return;
    }
    for bad in [0usize, 1, 21, 100] {
        assert!(
            V3Engine::load_with_depth(FULL, bad).is_err(),
            "depth {bad} should be refused"
        );
    }
    println!("depths outside 2..=20 are refused");
}
