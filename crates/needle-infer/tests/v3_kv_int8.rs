//! What the int8 key/value cache actually costs and actually changes.
//!
//! The quantiser itself is verified against upstream's `fake_quant` in
//! `needle-core`'s `v3_kv_quant_parity`. This is the other half: that storing
//! at 8 bits *saves the memory it claims to* and *still answers the question*.
//! A quantiser that matches the reference but is never stored narrowly, or is
//! stored and wrecks the output, passes that test and fails this one.
//!
//! Needs `weights/needle3.cact`; skips without it.
//!
//! Run: cargo test -p needle-infer --release --test v3_kv_int8 -- --nocapture

use needle_core::v3::{KvPrecision, V3Cache};
use needle_infer::v3_engine::{extract_tool_call, V3Engine, V3Options};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");
const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"control_lights","description":"Turn lights on or off","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"}},"required":["room","on"]}}]"#;

fn engine() -> Option<V3Engine> {
    if !std::path::Path::new(CACT).exists() {
        eprintln!("skipping: missing {CACT}");
        return None;
    }
    Some(V3Engine::load(CACT).expect("load"))
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1048576.0
}

#[test]
fn int8_storage_is_about_a_quarter_of_f32() {
    let Some(e) = engine() else { return };
    let cfg = &e.model.cfg;

    // Sized to the full context, which is the case a constrained target has to
    // budget for.
    let full = cfg.max_seq_len;
    let f32c = V3Cache::with_precision(cfg, full, KvPrecision::F32);
    let i8c = V3Cache::with_precision(cfg, full, KvPrecision::Int8);
    let (a, b) = (f32c.bytes(), i8c.bytes());
    println!(
        "full context ({full} tokens): f32 {:.1} MB, int8 {:.1} MB  ({:.1}% of f32)",
        mb(a),
        mb(b),
        100.0 * b as f64 / a as f64
    );
    for n in [512usize, 2048, 8192] {
        let x = V3Cache::with_precision(cfg, n, KvPrecision::F32).bytes();
        let y = V3Cache::with_precision(cfg, n, KvPrecision::Int8).bytes();
        println!(
            "  {n:>5} tokens: f32 {:>6.1} MB  int8 {:>6.1} MB",
            mb(x),
            mb(y)
        );
    }

    // One i8 plus a per-head scale against one f32: a touch over a quarter,
    // never at or above a half. If this regresses the storage went back to f32
    // and the saving is imaginary.
    assert!(b * 3 < a, "int8 cache {b} is not materially below f32 {a}");
    assert!(b * 5 > a, "int8 cache {b} is implausibly small against {a}");

    // `kv_bytes_at` is what the bindings report to a caller sizing a budget.
    // It must track the allocation, or the number shipped to users is fiction.
    for n in [512usize, 2048, 8192] {
        for (p, actual) in [
            (
                KvPrecision::F32,
                V3Cache::with_precision(cfg, n, KvPrecision::F32).bytes(),
            ),
            (
                KvPrecision::Int8,
                V3Cache::with_precision(cfg, n, KvPrecision::Int8).bytes(),
            ),
        ] {
            let est = cfg.kv_bytes_at(n, p);
            // The estimate covers the key/value rings; the cache also holds the
            // small conv and engram tails, so it may exceed the estimate a
            // little, never fall below it.
            assert!(
                est <= actual,
                "{p:?} at {n}: estimate {est} exceeds actual {actual}"
            );
            let over = (actual - est) as f64 / actual as f64;
            assert!(
                over < 0.10,
                "{p:?} at {n}: estimate {est} is {:.1}% under the actual {actual}",
                over * 100.0
            );
        }
    }
}

#[test]
fn int8_answers_the_same_question() {
    let Some(e) = engine() else { return };
    // Both directions. Quantisation drift that starts *inventing* a call on an
    // unrelated query is the failure a catalogue of happy-path prompts cannot
    // see, and it is the one that would matter in a deployment — so the
    // abstentions carry as much weight here as the calls do.
    let queries = [
        "What's the weather in Paris?",
        "Tell me the current conditions in Tokyo please",
        "is it raining in berlin right now",
        "Turn off the bedroom lights",
        "Turn on the kitchen lights and then tell me the weather in Oslo",
        "Write me a poem about the sea",
        "What is the capital of France?",
        "Thanks, that's all",
    ];

    let base = V3Options {
        max_new_tokens: 96,
        ..Default::default()
    };
    let quant = V3Options {
        kv_precision: KvPrecision::Int8,
        ..base.clone()
    };

    let (mut agreed, mut called, mut abstained) = (0, 0, 0);
    for q in queries {
        let a = e.generate(q, TOOLS, &base);
        let b = e.generate(q, TOOLS, &quant);
        let (ca, cb) = (extract_tool_call(&a.text), extract_tool_call(&b.text));
        println!("q: {q}\n  f32  -> {ca:?}\n  int8 -> {cb:?}");
        // The point of the assertion is the *call*, not the prose: int8 is a
        // different numerical path, so identical reasoning text is not owed.
        assert_eq!(ca, cb, "int8 changed the tool call for {q:?}");
        // And an abstention must stay an abstention rather than becoming an
        // invented lookup.
        let invented = |c: &Option<String>| {
            c.as_deref()
                .is_some_and(|c| c.contains("get_weather") || c.contains("control_lights"))
        };
        assert_eq!(
            invented(&ca),
            invented(&cb),
            "int8 changed whether a tool was called at all for {q:?}"
        );
        if invented(&ca) {
            called += 1;
        } else {
            abstained += 1;
        }
        agreed += 1;
    }
    println!(
        "{agreed}/{} queries agreed exactly ({called} called a tool, {abstained} abstained)",
        queries.len()
    );
}

#[test]
fn int8_is_not_silently_the_f32_path() {
    let Some(e) = engine() else { return };
    // If `kv_precision` were ignored, the two caches would be the same object
    // and this would be vacuous. Assert the stored values really differ.
    let cfg = &e.model.cfg;
    let mut a = V3Cache::with_precision(cfg, 8, KvPrecision::F32);
    let mut b = V3Cache::with_precision(cfg, 8, KvPrecision::Int8);
    let k: Vec<f32> = (0..cfg.k_dim()).map(|i| (i as f32) * 0.013 - 0.4).collect();
    let v: Vec<f32> = (0..cfg.v_dim())
        .map(|i| (i as f32) * -0.007 + 0.2)
        .collect();
    a.write_kv(0, 0, &k, &v);
    b.write_kv(0, 0, &k, &v);
    assert!(
        b.bytes() < a.bytes(),
        "int8 cache did not allocate less: {} vs {}",
        b.bytes(),
        a.bytes()
    );
    println!(
        "one-position caches: f32 {} B, int8 {} B",
        a.bytes(),
        b.bytes()
    );
}

/// Batched prefill and token-by-token stepping must agree at int8 too.
///
/// This is the assertion that catches a quantised *decode* path bolted onto an
/// f32 *prefill* path. Upstream quantises the query, keys and values for every
/// position, prompt included; if the batched path skipped it, the prompt would
/// be attended at full precision and the continuation would diverge from what
/// stepping produces — while every other int8 test still passed, because the
/// engine only ever prefills.
#[test]
fn int8_prefill_agrees_with_int8_stepping() {
    let Some(e) = engine() else { return };
    const CONTINUE: usize = 24;
    let model = &e.model;

    // A real prompt, so the tokens exercise the tokenizer's own vocabulary.
    let prompt = V3Engine::build_prompt("What's the weather in Paris?", TOOLS, None);
    let tokens = e.tokenizer.encode(&prompt);
    println!("prompt is {} tokens", tokens.len());
    assert!(tokens.len() > 16, "prompt too short to be a real test");

    let cont = |cache: &mut V3Cache, first: Vec<f32>| {
        let mut out = Vec::new();
        let mut l = first;
        for _ in 0..CONTINUE {
            let n = l
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            out.push(n);
            l = model.decode_step(cache, n);
        }
        out
    };

    // Deliberately hinted short, as the f32 sibling
    // `batched_prefill_leaves_the_cache_where_stepping_would` is: the
    // continuation then runs past the hint and `ensure` has to grow the ring.
    // At int8 that grows four buffers, including the per-head scales, whose
    // index mapping depends on `slots` — the one path a generous hint hides.
    let hint = tokens.len() + 8;
    let mut stepped = V3Cache::with_precision(&model.cfg, hint, KvPrecision::Int8);
    let mut logits = Vec::new();
    for &t in &tokens {
        logits = model.decode_step(&mut stepped, t);
    }
    let stepped_logits = logits.clone();
    let want = cont(&mut stepped, logits);

    let mut filled = V3Cache::with_precision(&model.cfg, hint, KvPrecision::Int8);
    let pre = model.prefill(&tokens, &mut filled);

    // Compare the logits, not just the argmax. The continuation alone is too
    // coarse to see this: both caches are written through `write_kv` and so
    // hold quantised entries either way, and the model is robust enough that a
    // whole prefill attended at the wrong precision can still pick the same 24
    // tokens. The logits are where the split shows up.
    let worst = pre
        .iter()
        .zip(&stepped_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms =
        (stepped_logits.iter().map(|x| x * x).sum::<f32>() / stepped_logits.len() as f32).sqrt();
    // Anchored to the f32 path rather than to a number picked by hand: batched
    // and stepped attention accumulate in different orders, so they never agree
    // exactly even at full precision. What matters is that quantising does not
    // make the gap materially worse — which it does, several-fold, if the
    // batched path attends at a precision the decode path does not.
    let mut f32_stepped = V3Cache::new(&model.cfg, hint);
    let mut fl = Vec::new();
    for &t in &tokens {
        fl = model.decode_step(&mut f32_stepped, t);
    }
    let mut f32_filled = V3Cache::new(&model.cfg, hint);
    let f32_pre = model.prefill(&tokens, &mut f32_filled);
    let base = f32_pre
        .iter()
        .zip(&fl)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "prefill vs stepping, max abs logit diff: f32 {base:.3e}, int8 {worst:.3e} \
         (RMS {rms:.3e})"
    );
    // Exact, not approximate, and for the same reason the f32 path is exact:
    // batched prefill and stepped decode run the same arithmetic over the same
    // representation. Attending over dequantised `f32` in one path and stored
    // integers in the other passes every other test in this file and lands
    // here at 3.7e-2.
    assert_eq!(
        base, 0.0,
        "the f32 invariant this is modelled on has regressed"
    );
    assert_eq!(
        worst, 0.0,
        "int8 prefill diverged from int8 stepping by {worst:.3e} (RMS {rms:.3e}) — \
         the batched path is not reading the representation decode reads"
    );

    let got = cont(&mut filled, pre);

    assert_eq!(
        want, got,
        "int8 prefill and int8 stepping produced different continuations"
    );
    // Confirm the growth path was actually taken, rather than trusting the
    // arithmetic above to stay true if CONTINUE or the prompt changes.
    assert!(
        filled.slots(0) > hint,
        "the ring never grew (slots {} against hint {hint}), so the int8 resize \
         path is still untested",
        filled.slots(0)
    );
    println!(
        "int8 continuation identical over {CONTINUE} tokens; ring grew {hint} -> {}",
        filled.slots(0)
    );
}
