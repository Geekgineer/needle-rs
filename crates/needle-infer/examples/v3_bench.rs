//! Repeatable single-query timing for the v3 path, as quoted in BENCHMARKS.md.
//!
//! Separates load, prefill and decode, and reports the int8 key/value cache
//! beside the f32 one — on v3 the cache is large enough that its cost is a
//! deployment decision, not a footnote.
//!
//! Run: cargo run --release -p needle-infer --example v3_bench

use needle_core::v3::{KvPrecision, V3Cache};
use needle_infer::cact::CactV3;
use needle_infer::v3::model_from_cact;
use needle_infer::v3_engine::{V3Engine, V3Options};
use std::time::Instant;

const TOOLS: &str = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}]"#;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/needle3.cact".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("missing {path}");
        std::process::exit(1);
    }
    let q = "What's the weather in Paris?";

    let t0 = Instant::now();
    let e = V3Engine::load(&path).unwrap();
    println!("load: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);

    // Prefill and decode, separated. A single tokens-per-second figure folds a
    // ~60-token prefill into the answer and understates decode several-fold.
    let cact = CactV3::load(&path).unwrap();
    let model = model_from_cact(&cact).unwrap();
    let prompt = V3Engine::build_prompt(q, TOOLS, None);
    let ids = e.tokenizer.encode(&prompt);
    println!("prompt: {} tokens", ids.len());

    for precision in [KvPrecision::F32, KvPrecision::Int8] {
        let mut best_pre = f64::INFINITY;
        let mut best_dec = f64::INFINITY;
        for _ in 0..3 {
            let mut cache = V3Cache::with_precision(&model.cfg, ids.len() + 64, precision);
            let t = Instant::now();
            let mut logits = model.prefill(&ids, &mut cache);
            let pre = t.elapsed().as_secs_f64() * 1e3;

            let t = Instant::now();
            const STEPS: usize = 32;
            for _ in 0..STEPS {
                let n = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap();
                logits = model.decode_step(&mut cache, n);
            }
            let dec = t.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

            best_pre = best_pre.min(pre);
            best_dec = best_dec.min(dec);
        }
        println!(
            "{precision:?}: prefill {:.2} ms ({:.2} ms/token), decode {:.2} ms/token, \
             cache {:.1} MB at 512 / {:.1} MB at full context",
            best_pre,
            best_pre / ids.len() as f64,
            best_dec,
            model.cfg.kv_bytes_at(512, precision) as f64 / 1048576.0,
            model.cfg.kv_bytes_at(model.cfg.max_seq_len, precision) as f64 / 1048576.0,
        );
    }

    // End to end through the engine, which is what a caller actually sees.
    for &max in &[32usize, 64, 128] {
        let opts = V3Options {
            max_new_tokens: max,
            ..Default::default()
        };
        let mut best = f64::INFINITY;
        let mut ttft = 0.0;
        let mut produced = 0;
        for _ in 0..3 {
            let mut first = 0.0;
            let t = Instant::now();
            let r = e.generate_with(q, TOOLS, &opts, |_, _| {
                if first == 0.0 {
                    first = t.elapsed().as_secs_f64() * 1e3;
                }
            });
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                ttft = first;
                produced = r.tokens.len();
            }
        }
        println!(
            "max_new_tokens {max:>4}: {best:>8.0} ms total, {ttft:>6.0} ms to first token, \
             {produced} produced"
        );
    }
}
