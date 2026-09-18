//! A `V3Engine` is shared by `&self`, so nothing a call does may outlive it.
//!
//! v2 covers this with four tests — `reset_yields_identical_second_run`,
//! `model_is_stateless_across_sequences`, `shared_state_matches_fresh_state`
//! and `head_runs_do_not_affect_generation`. v3 had none, and it has *more*
//! state to leak than v2 did: a per-session `V3Cache` holding three separate
//! histories (key/value rings, the raw Q/K/V convolution tails and the Engram
//! token and value tails), plus a probe head pooled across positions. A tail
//! left seeded from a previous call would not fail loudly; it would produce a
//! few plausible tokens and then drift, which is exactly the failure a
//! generation test with one prompt cannot see.
//!
//! Needs `weights/needle3.cact`; skips without it.
//!
//! Run: cargo test -p needle-infer --release --test v3_statelessness -- --nocapture

use needle_infer::v3_engine::{V3Engine, V3Options};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"control_lights","description":"Turn lights on or off","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}}]"#;

fn engine() -> Option<V3Engine> {
    if !std::path::Path::new(CACT).exists() {
        eprintln!("skipping: missing {CACT}");
        return None;
    }
    Some(V3Engine::load(CACT).expect("load"))
}

fn opts() -> V3Options {
    V3Options {
        max_new_tokens: 96,
        ..Default::default()
    }
}

#[test]
fn a_second_run_reproduces_the_first_exactly() {
    let Some(e) = engine() else { return };
    let q = "What's the weather in Paris?";
    let a = e.generate(q, TOOLS, &opts());
    let b = e.generate(q, TOOLS, &opts());
    assert_eq!(
        a.tokens, b.tokens,
        "the same prompt twice on one engine diverged:\n  1: {:?}\n  2: {:?}",
        a.text, b.text
    );
    println!("repeat run identical over {} tokens", a.tokens.len());
}

#[test]
fn an_intervening_run_does_not_change_the_answer() {
    let Some(e) = engine() else { return };
    // Different length, different tools exercised, different Engram n-grams —
    // so a surviving convolution or token tail would carry across.
    let q = "What's the weather in Paris?";
    let other = "Turn off the bedroom lights and tell me if it is raining in Reykjavik";

    let clean = e.generate(q, TOOLS, &opts());
    let _ = e.generate(other, TOOLS, &opts());
    let after = e.generate(q, TOOLS, &opts());

    assert_eq!(
        clean.tokens, after.tokens,
        "a previous generation leaked into the next:\n  clean {:?}\n  after {:?}",
        clean.text, after.text
    );
    println!(
        "unaffected by an intervening generation of {} tokens",
        other.len()
    );
}

#[test]
fn a_fresh_engine_agrees_with_a_reused_one() {
    let Some(e) = engine() else { return };
    let q = "Turn on the kitchen lights";
    // Warm the shared engine with unrelated work first.
    let _ = e.generate("What's the weather in Tokyo?", TOOLS, &opts());
    let reused = e.generate(q, TOOLS, &opts());

    let fresh_engine = engine().expect("second load");
    let fresh = fresh_engine.generate(q, TOOLS, &opts());

    assert_eq!(
        fresh.tokens, reused.tokens,
        "a reused engine disagreed with a freshly loaded one:\n  fresh  {:?}\n  reused {:?}",
        fresh.text, reused.text
    );
    println!("reused engine matches a fresh load");
}

#[test]
fn running_the_confidence_head_does_not_perturb_generation() {
    let Some(e) = engine() else { return };
    let q = "What's the weather in Paris?";
    let before = e.generate(q, TOOLS, &opts());

    // The head pools over every position and shares the model's weights. If it
    // wrote through any of that, this is where it shows.
    let completion =
        r#"<tool_call>[{"name":"get_weather","arguments":{"city":"Paris"}}]</tool_call>"#;
    let p1 = e
        .confidence_for(q, TOOLS, completion)
        .expect("confidence head");
    let after = e.generate(q, TOOLS, &opts());
    let p2 = e
        .confidence_for(q, TOOLS, completion)
        .expect("confidence head");

    assert_eq!(
        before.tokens, after.tokens,
        "scoring a completion changed the next generation"
    );
    assert_eq!(p1, p2, "the head disagreed with itself across a generation");
    println!("confidence {p1:.6} stable across a generation");
}

#[test]
fn a_long_prompt_does_not_contaminate_a_short_one() {
    let Some(e) = engine() else { return };
    // The Engram value convolution reaches nine positions back and the Q/K/V
    // convolution two. A long run followed by a short one is the ordering that
    // exposes a tail which was seeded but never cleared.
    let short = "Weather in Oslo?";
    let long = "I have just landed in Paris after a very long flight from Sydney \
                with a stopover in Singapore, and before I decide what to wear \
                this evening I would like to know what the weather is doing there";

    let clean = e.generate(short, TOOLS, &opts());
    let _ = e.generate(long, TOOLS, &opts());
    let after = e.generate(short, TOOLS, &opts());

    assert_eq!(
        clean.tokens, after.tokens,
        "a long prompt left state behind:\n  clean {:?}\n  after {:?}",
        clean.text, after.text
    );
    println!("short prompt unaffected by a preceding long one");
}
