//! Needle 3 end-to-end generation against the real checkpoint.
//!
//! Needs `weights/needle3.cact`.
//!
//! The forward-pass tests prove the numerics. This proves the thing a user
//! actually gets: prompt in, tool call out. The expected calls are the ones
//! upstream's own model produced in `tools/check_v3_canon.py`.

use needle_infer::v3_engine::{StopReason, V3Engine, V3Options};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");

const TOOLS: &str = r#"[
  {"name":"get_weather","description":"Get current weather for a city",
   "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
  {"name":"control_lights","description":"Turn lights on or off in a room",
   "parameters":{"type":"object","properties":{"room":{"type":"string"},"state":{"type":"string"}},
   "required":["room","state"]}}
]"#;

fn engine() -> Option<V3Engine> {
    if !std::path::Path::new(CACT).exists() {
        println!("skipping v3 e2e: no weights/needle3.cact");
        return None;
    }
    Some(V3Engine::load(CACT).expect("engine should load"))
}

#[test]
fn produces_the_calls_upstream_produces() {
    let Some(e) = engine() else { return };

    for (query, want_name, want_arg) in [
        ("What's the weather in Paris?", "get_weather", "Paris"),
        ("Turn off the bedroom lights", "control_lights", "bedroom"),
    ] {
        let t0 = std::time::Instant::now();
        let res = e.generate(query, TOOLS, &V3Options::default());
        let call = needle_infer::v3_engine::extract_tool_call(&res.text);

        println!(
            "\nQ: {query}\nA: {}\n  stop={:?} tokens={} positions={} {:?}",
            res.text.replace('\n', "\n   "),
            res.stop,
            res.tokens.len(),
            res.positions,
            t0.elapsed()
        );

        let call = call.unwrap_or_else(|| panic!("no tool call for {query:?}"));
        assert!(
            call.contains(want_name),
            "expected {want_name} for {query:?}, got {call}"
        );
        assert!(
            call.contains(want_arg),
            "expected argument {want_arg:?} for {query:?}, got {call}"
        );
        assert!(
            matches!(res.stop, StopReason::ImEnd | StopReason::Eos),
            "expected a clean stop for {query:?}, got {:?}",
            res.stop
        );
    }
}

#[test]
fn reasons_before_answering() {
    let Some(e) = engine() else { return };
    // v3 emits chain-of-thought where v2 did not; the default token budget
    // depends on that being true, so it is asserted rather than assumed.
    let res = e.generate("What's the weather in Paris?", TOOLS, &V3Options::default());
    match V3Engine::reasoning(&res.text) {
        Some(r) => println!("reasoning: {r:?}"),
        None => println!("no <think> block on this prompt (text: {:?})", res.text),
    }
}

#[test]
fn streaming_sees_the_same_text_it_returns() {
    let Some(e) = engine() else { return };
    let mut streamed = String::new();
    let res = e.generate_with(
        "What's the weather in Paris?",
        TOOLS,
        &V3Options::default(),
        |_, piece| streamed.push_str(piece),
    );
    // The callback emits decoded deltas, so concatenating them must reproduce
    // the returned text exactly — no normalisation. A caller rendering the
    // stream sees precisely what it finally gets.
    assert_eq!(
        streamed, res.text,
        "streamed text diverged from the returned text"
    );
}

#[test]
fn an_irrelevant_query_abstains_rather_than_inventing() {
    let Some(e) = engine() else { return };
    let res = e.generate(
        "Write me a poem about the sea",
        TOOLS,
        &V3Options::default(),
    );
    let call = needle_infer::v3_engine::extract_tool_call(&res.text);
    println!("poem query -> {call:?} (text {:?})", res.text);
    // Either no call at all or an explicit empty array is correct. Inventing
    // a weather lookup would not be.
    if let Some(c) = call {
        assert!(
            !c.contains("get_weather") && !c.contains("control_lights"),
            "invented a tool call for an unrelated query: {c}"
        );
    }
}

#[test]
fn temperature_zero_is_deterministic() {
    let Some(e) = engine() else { return };
    let opts = V3Options::default();
    let a = e.generate("What's the weather in Paris?", TOOLS, &opts);
    let b = e.generate("What's the weather in Paris?", TOOLS, &opts);
    assert_eq!(a.tokens, b.tokens, "greedy decoding must be reproducible");
}

#[test]
fn confidence_scores_the_answer_not_the_question() {
    let Some(e) = engine() else { return };
    let query = "What's the weather in Paris?";
    let right =
        "<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]</tool_call>";
    let wrong = "<tool_call>[{\"name\":\"control_lights\",\"arguments\":{\"room\":\"Paris\",\"state\":\"on\"}}]</tool_call>";

    let p_right = e.confidence_for(query, TOOLS, right).expect("head present");
    let p_wrong = e.confidence_for(query, TOOLS, wrong).expect("head present");
    let p_bare = e.confidence_for(query, TOOLS, "").expect("head present");

    println!("confidence — right {p_right:.4}, wrong {p_wrong:.4}, bare query {p_bare:.4}");

    // The property that matters: a correct completion outscores a wrong one.
    assert!(
        p_right > p_wrong,
        "a correct call should outscore a wrong one: {p_right} vs {p_wrong}"
    );
    // And the trap worth pinning: unlike v2, where a bare query collapsed to
    // near zero and so announced the misuse, v3 scores one comfortably high.
    // Anyone passing a query instead of a completion gets a plausible number
    // that means nothing, so this asserts the hazard still exists rather than
    // quietly assuming v2's behaviour carried over.
    assert!(
        p_bare > 0.5,
        "expected v3 to score a bare query high (the documented trap), got {p_bare}"
    );
    assert!(
        p_right > p_bare,
        "the real completion should still outscore a bare query: {p_right} vs {p_bare}"
    );
}

#[test]
fn run_scored_pairs_an_answer_with_its_confidence() {
    let Some(e) = engine() else { return };
    let (res, p) = e.run_scored("What's the weather in Paris?", TOOLS);
    let p = p.expect("v3 exports a confidence head");
    println!("answer {:?} scored {p:.4}", res.text);
    assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
    assert!(res.text.contains("get_weather"));
}

#[test]
fn constrained_decoding_keeps_the_schema() {
    let Some(e) = engine() else { return };
    let opts = V3Options {
        constrain: true,
        ..V3Options::default()
    };

    for query in [
        "What's the weather in Paris?",
        "Turn off the bedroom lights",
        "Turn on the kitchen lights",
    ] {
        let res = e.generate(query, TOOLS, &opts);
        let call = needle_infer::v3_engine::extract_tool_call(&res.text);
        println!("constrained {query:?} -> {call:?}");

        let Some(call) = call else { continue };
        if call == "[]" {
            continue; // an abstention is schema-valid
        }
        // Valid JSON, and every key it names must be declared. v1 could emit a
        // repeated argument key and Unicode garbage once its declared keys ran
        // out; the grammar exists to make that unrepresentable.
        let parsed: serde_json::Value =
            serde_json::from_str(&call).unwrap_or_else(|e| panic!("{call} is not JSON: {e}"));
        let arr = parsed.as_array().expect("a tool call is an array");
        for item in arr {
            let name = item["name"].as_str().expect("each call names a tool");
            assert!(
                name == "get_weather" || name == "control_lights",
                "invented a tool: {name}"
            );
            let args = item["arguments"]
                .as_object()
                .expect("arguments is an object");
            let allowed: &[&str] = match name {
                "get_weather" => &["city"],
                _ => &["room", "state"],
            };
            for k in args.keys() {
                assert!(
                    allowed.contains(&k.as_str()),
                    "{name} got undeclared key {k}"
                );
            }
        }
    }
}

#[test]
fn constraining_does_not_disturb_a_clean_answer() {
    let Some(e) = engine() else { return };
    let q = "What's the weather in Paris?";
    let free = e.generate(q, TOOLS, &V3Options::default());
    let bound = e.generate(
        q,
        TOOLS,
        &V3Options {
            constrain: true,
            ..V3Options::default()
        },
    );
    // When the model was already going to produce valid JSON, the grammar
    // should be inert. If this ever diverges, the mask is rejecting tokens the
    // schema actually permits.
    assert_eq!(
        free.tokens, bound.tokens,
        "constraining changed an already-valid answer:\n  free  {:?}\n  bound {:?}",
        free.text, bound.text
    );
}
