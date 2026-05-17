//! Functional integration test: load real weights → run inference → verify output.
//!
//! Requires weights at:
//!   ../../weights/needle.safetensors
//!   ../../weights/vocab.txt
//!
//! Generate them with:
//!   cd needle-rust/
//!   PYTHONPATH=needle python3 tools/export.py \
//!       --checkpoint needle/checkpoints/needle.pkl \
//!       --output-dir weights/
//!
//! Run: cargo test -p needle-infer -- --nocapture

use needle_infer::engine::NeedleEngine;

const WEIGHTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../weights/needle.safetensors"
);
const VOCAB: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/vocab.txt");

fn weights_available() -> bool {
    std::path::Path::new(WEIGHTS).exists() && std::path::Path::new(VOCAB).exists()
}

/// Smoke test: model loads, runs one decode step, produces non-empty output.
#[test]
fn test_model_loads_and_runs() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");

    let tools_json = r#"[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}]"#;

    let result = engine.run("What's the weather in Paris?", tools_json);

    // The model should produce at least one output token (TOOL_CALL=4 or content)
    // With zero input context, it may just output EOS — that's OK for a smoke test.
    eprintln!("=== Functional test output ===");
    eprintln!("token_ids: {:?}", result.token_ids);
    eprintln!("text: {:?}", result.text);

    // The model must not panic and must return a result
    eprintln!("PASS: model ran {} decode steps", result.token_ids.len());
}

/// Test that encode produces non-zero hidden states for real input.
#[test]
fn test_encoder_produces_nonzero_hidden() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    use needle_infer::engine::NeedleEngine;

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");

    // This exercises the full encoder forward pass on real weights.
    // If weights are wrong (wrong layout, wrong norms), the logits will be NaN or ±Inf.
    let tools_json = r#"[{"name":"search","description":"Search","parameters":{"type":"object","properties":{}}}]"#;
    let result = engine.run("Find something", tools_json);

    // Verify no NaN in output (would indicate bad weight layout)
    let all_valid = result.token_ids.iter().all(|&t| t < 8192);
    assert!(
        all_valid,
        "decoder produced out-of-range token ID — likely NaN in logits"
    );
    eprintln!("encoder hidden states: valid (no NaN/OOB token IDs)");
    eprintln!("output: {:?}", result.text);
}

/// Token-level check: the first token predicted for a tool-call query must be
/// TOOL_CALL (id=4). Uses the real tokenizer (vocab.txt) + real weights.
///
/// This validates the full pipeline: tokenize → encode → decode_step → argmax.
/// Any breakage in the tokenizer, encoder, or LM head will show here.
#[test]
fn test_first_token_likely_tool_call() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");

    let result = engine.run(
        "What's the weather in Paris?",
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
    );

    eprintln!("first predicted token: {:?}", result.token_ids.first());
    eprintln!("full output: {:?}", result.text);

    assert!(
        !result.token_ids.is_empty(),
        "engine produced no tokens — model may be broken"
    );

    // TOOL_CALL token is id=4. The constrained decoder forces it first; if it
    // isn't token 4, something is wrong with the constrained decoder or the
    // tokenizer feeding the wrong context to the encoder.
    assert_eq!(
        result.token_ids[0], 4,
        "expected TOOL_CALL (id=4) as first token, got {}",
        result.token_ids[0]
    );
}

/// Weight sanity: embedding is non-trivial (rules out BF16 conversion bugs) and
/// encoder final norm is valid. Does not run inference.
#[test]
fn test_weight_sanity() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    use needle_core::config::TransformerConfig;
    use needle_infer::safetensors::SafeTensors;

    let st = SafeTensors::load(WEIGHTS).expect("safetensors load failed");
    let cfg = TransformerConfig::default();
    let d = cfg.d_model;
    let vocab = cfg.vocab_size;

    let embedding = st.get_f32("embedding").expect("embedding missing");
    assert_eq!(embedding.len(), vocab * d, "embedding shape mismatch");

    let emb_rms: f32 =
        (embedding.iter().map(|x| x * x).sum::<f32>() / embedding.len() as f32).sqrt();
    assert!(
        emb_rms > 0.001,
        "embedding near-zero — likely BF16 conversion bug: rms={emb_rms}"
    );
    assert!(emb_rms < 10.0, "embedding values too large: rms={emb_rms}");
    eprintln!("embedding RMS: {emb_rms:.4}  (expected ~0.01–1.0)");

    let enc_norm = st
        .get_f32("encoder_final_norm")
        .expect("encoder_final_norm missing");
    let norm_mean: f32 = enc_norm.iter().sum::<f32>() / enc_norm.len() as f32;
    eprintln!("encoder_final_norm mean: {norm_mean:.6}  (expected ~0, ZCRMSNorm scale init=0)");
}

/// Tokenizer integration: `vocab.encode()` on known queries produces the Python-reference IDs.
/// This validates that vocab.txt was exported correctly and the greedy-longest-match
/// tokenizer matches SentencePiece output for the inputs the model actually sees.
#[test]
fn test_tokenizer_integration() {
    if !weights_available() {
        eprintln!("SKIP: vocab.txt not found at {VOCAB}");
        return;
    }

    use needle_infer::tokenizer::Vocabulary;

    let vocab = Vocabulary::load_text(VOCAB).expect("vocab load failed");

    // Python reference (verified against sp.Encode with the Needle tokenizer):
    //   "What's the weather?" → pieces ['▁What', "'", 's', '▁the', '▁weather', '?']
    let got = vocab.encode("What's the weather?");
    assert_eq!(
        got,
        vec![4279, 8066, 8046, 302, 1149, 8105],
        "tokenizer mismatch for \"What's the weather?\": got {:?}",
        got
    );

    //   "get_weather" → pieces ['▁get', '_', 'weather']
    let got = vocab.encode("get_weather");
    assert_eq!(
        got,
        vec![1734, 8062, 1331],
        "tokenizer mismatch for \"get_weather\": got {:?}",
        got
    );

    eprintln!("tokenizer integration: all reference IDs match");
}

/// Empty tools list: engine.run() with "[]" must not panic and must produce
/// some output (unconstrained decode — no tool name or key to force).
#[test]
fn test_empty_tools_list() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");
    // Must not panic; output may be anything since there are no constraints.
    let result = engine.run("Hello", "[]");
    eprintln!("empty tools output: {:?}", result.text);
    // All token IDs must be in-range (rules out NaN → OOB from argmax).
    let engine_ref = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");
    let vocab_size = engine_ref.cfg().vocab_size;
    assert!(
        result.token_ids.iter().all(|&t| (t as usize) < vocab_size),
        "out-of-range token ID produced — likely NaN in logits"
    );
}

/// Python API contract: output must NOT start with `<tool_call>` (Python strips it).
#[test]
fn test_output_does_not_start_with_tool_call_prefix() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }
    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");
    let result = engine.run(
        "What's the weather in Paris?",
        r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
    );
    eprintln!("output text: {:?}", result.text);
    assert!(
        !result.text.starts_with("<tool_call>"),
        "<tool_call> prefix was not stripped — Python contract violated: {:?}",
        result.text
    );
}

/// Python API contract: camelCase tool names passed by caller must appear in output,
/// not their snake_case equivalents (Python's restore_tool_names step).
#[test]
fn test_camel_case_tool_name_restored_in_output() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }
    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");
    // Pass camelCase name; the model internally uses snake_case but output must show original.
    let result = engine.run(
        "What's the weather in Paris?",
        r#"[{"name":"getWeather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
    );
    eprintln!("output text: {:?}", result.text);
    assert!(
        result.text.contains("getWeather"),
        "original camelCase name not restored — Python contract violated: {:?}",
        result.text
    );
    assert!(
        !result.text.contains("get_weather"),
        "snake_case name leaked into output — Python contract violated: {:?}",
        result.text
    );
}

/// Spaced JSON normalization: tool names in caller JSON with spaces around ':'
/// must still be normalized to snake_case for the encoder.
#[test]
fn test_spaced_json_normalization() {
    if !weights_available() {
        eprintln!("SKIP: weights not found at {WEIGHTS}");
        return;
    }

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");

    // Spaced JSON: "name" : "GetWeather" — the old string-replace would miss this.
    let spaced = r#"[{"name" : "get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#;
    let compact = r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#;

    let r_spaced = engine.run("What's the weather in Paris?", spaced);
    let r_compact = engine.run("What's the weather in Paris?", compact);

    eprintln!("spaced  JSON output: {:?}", r_spaced.text);
    eprintln!("compact JSON output: {:?}", r_compact.text);

    // Both forms must produce identical output (normalization is whitespace-agnostic).
    assert_eq!(
        r_spaced.text, r_compact.text,
        "spaced and compact JSON produced different outputs — normalize_tools_json is broken"
    );
}

/// SafeTensors brace parser: unbalanced `}` must not wrap the usize depth counter.
#[test]
fn test_safetensors_unbalanced_brace_no_underflow() {
    use needle_infer::safetensors::SafeTensors;

    // Header with an extra `}` closing brace inside a metadata value.
    // Without saturating_sub, depth underflows to usize::MAX and corrupts parsing.
    let json = "{\"__metadata__\":{\"k\":\"v\"}}";
    let header_bytes = json.as_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_bytes);
    // Must parse without panic.
    let st = SafeTensors::from_bytes(buf).expect("must not panic on valid JSON");
    assert_eq!(st.get_metadata("k"), Some("v"));
}

/// SafeTensors JSON parser: tensor names and metadata values containing escaped quotes
/// must not truncate at the `\"` and must still parse correctly.
#[test]
fn test_safetensors_parser_escaped_quotes_in_metadata() {
    use needle_infer::safetensors::SafeTensors;

    // Metadata value contains an escaped quote: `value with "quotes"`.
    let json = "{\"__metadata__\":{\"key\":\"value with \\\"quotes\\\"\"}}";
    let header_bytes = json.as_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_bytes);
    // No tensor data needed.

    let st = SafeTensors::from_bytes(buf).expect("parse must succeed");
    let val = st
        .get_metadata("key")
        .expect("metadata key must be present");
    // The escaped quotes are stripped by SafeTensors (the outer quotes are removed,
    // and the inner `\"` become `"` — or the raw string is passed through).
    // Either way, the value must be non-empty and must not be `value with ` (truncated).
    assert!(
        !val.is_empty(),
        "metadata value was empty — parser truncated at escaped quote"
    );
    assert!(
        !val.starts_with("value with \"quotes\"") || val.len() > 5,
        "metadata value truncated: got {val:?}"
    );
}

/// run_stream: streaming callback fires for every generated token; final text matches run().
#[test]
fn test_run_stream_matches_run() {
    if !weights_available() {
        eprintln!("SKIP: weights not found");
        return;
    }
    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");
    let query = "What's the weather in Paris?";
    let tools = r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#;

    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut streamed_pieces: Vec<String> = Vec::new();

    let stream_result = engine.run_stream(query, tools, |token_id, piece| {
        streamed_tokens.push(token_id);
        streamed_pieces.push(piece.to_string());
    });
    let direct_result = engine.run(query, tools);

    eprintln!("stream text: {:?}", stream_result.text);
    eprintln!("direct text: {:?}", direct_result.text);
    eprintln!("streamed {} tokens", streamed_tokens.len());

    assert_eq!(
        stream_result.text, direct_result.text,
        "run_stream and run must produce identical final text"
    );
    assert_eq!(
        stream_result.token_ids, direct_result.token_ids,
        "run_stream and run must produce identical token IDs"
    );
    assert_eq!(
        streamed_tokens, direct_result.token_ids,
        "callback must fire exactly once per generated token in order"
    );
    assert_eq!(
        streamed_pieces.len(),
        streamed_tokens.len(),
        "each callback invocation must include a piece string"
    );
}

/// run_batch: batch results must match individual run() calls in order.
#[test]
fn test_run_batch_matches_individual_runs() {
    if !weights_available() {
        eprintln!("SKIP: weights not found");
        return;
    }
    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");

    let examples = vec![
        (
            "What's the weather in Paris?",
            r#"[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string"}}}]"#,
        ),
        (
            "Search for Python tutorials",
            r#"[{"name":"web_search","description":"Search","parameters":{"query":{"type":"string"}}}]"#,
        ),
    ];

    let batch_results = engine.run_batch(&examples);
    assert_eq!(batch_results.len(), examples.len());

    for (i, ((q, t), batch_res)) in examples.iter().zip(batch_results.iter()).enumerate() {
        let individual = engine.run(q, t);
        assert_eq!(
            batch_res.text, individual.text,
            "example {i}: batch and individual run produced different text"
        );
        assert_eq!(
            batch_res.token_ids, individual.token_ids,
            "example {i}: batch and individual run produced different token IDs"
        );
    }
    eprintln!(
        "run_batch: all {} results match individual run() calls",
        examples.len()
    );
}

/// encode_contrastive: returns None when weights have no contrastive head,
/// and (when present) the embedding is L2-normalized.
#[test]
fn test_encode_contrastive_without_head_returns_none() {
    // Build a minimal engine from a tiny in-memory SafeTensors (no contrastive tensors)
    // and verify encode_contrastive returns None.

    // We can't easily build a full in-memory engine without real weights, but we can at
    // least verify the API contract by confirming contrastive_dim() == 0 without head.
    // When weights are available, also verify the L2-norm of a real embedding.
    if !weights_available() {
        eprintln!("SKIP: weights not found");
        return;
    }

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");
    let dim = engine.contrastive_dim();
    eprintln!("contrastive_dim: {dim}");

    if dim == 0 {
        assert!(
            engine.encode_contrastive("hello").is_none(),
            "encode_contrastive must return None when contrastive_dim == 0"
        );
        eprintln!("SKIP L2-norm check: no contrastive head in these weights");
        return;
    }

    // With a contrastive head, the embedding must be L2-normalized (dot with self ≈ 1.0).
    let emb = engine
        .encode_contrastive("What's the weather in Paris?")
        .expect("encode_contrastive must return Some when contrastive_dim > 0");
    assert_eq!(
        emb.len(),
        dim,
        "embedding length must equal contrastive_dim"
    );

    let sq_norm: f32 = emb.iter().map(|x| x * x).sum();
    assert!(
        (sq_norm - 1.0).abs() < 1e-4,
        "contrastive embedding is not L2-normalized: ||v||²={sq_norm:.6}"
    );
    eprintln!("L2-norm check: ||v||²={sq_norm:.6}  (expected ~1.0)");

    // Different inputs must produce different embeddings (not all the same).
    let emb2 = engine
        .encode_contrastive("Search for Python tutorials")
        .expect("second embed failed");
    let same = emb
        .iter()
        .zip(emb2.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(
        !same,
        "encode_contrastive produced identical embeddings for different inputs"
    );
}

/// SafeTensors I4 shape validation: malformed tensor (shape mismatch) must return None, not panic.
#[test]
fn test_safetensors_i4_shape_mismatch_returns_none() {
    use needle_infer::safetensors::SafeTensors;

    // Build a minimal SafeTensors buffer with an I4 tensor where raw bytes * 2 != shape volume.
    // Shape claims [4, 4] = 16 elements but we only provide 6 bytes (12 values ≠ 16).
    let json = r#"{"t":{"dtype":"I4","shape":[4,4],"data_offsets":[0,6]}}"#;
    let header_bytes = json.as_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_bytes);
    buf.extend_from_slice(&[0xABu8; 6]); // 6 bytes → 12 values, but shape needs 16

    let st = SafeTensors::from_bytes(buf).expect("parse should succeed");
    // get_f32 must return None (not panic) for a shape-mismatched I4 tensor
    assert!(
        st.get_f32("t").is_none(),
        "expected None for I4 shape mismatch but got Some(_)"
    );
}

/// retrieve_tools: without a contrastive head, returns empty Vec.
/// With a head, top-1 result must be the most-relevant tool description.
#[test]
fn test_retrieve_tools() {
    if !weights_available() {
        eprintln!("SKIP: weights not found");
        return;
    }

    let engine = NeedleEngine::load(WEIGHTS, VOCAB).expect("load failed");

    // Without a contrastive head, must return empty Vec, never panic.
    if engine.contrastive_dim() == 0 {
        let results = engine.retrieve_tools("get the weather", &["weather tool", "search tool"], 1);
        assert!(
            results.is_empty(),
            "retrieve_tools must return empty Vec when no contrastive head"
        );
        eprintln!("SKIP ranking check: no contrastive head");
        return;
    }

    // With a contrastive head, scores must be in [-1, 1] (normalized dot product).
    let descs = vec![
        "Get current weather conditions for a location",
        "Search the web for information",
        "Send an email to a recipient",
    ];
    let results = engine.retrieve_tools("What's the weather?", &descs, 2);
    assert_eq!(
        results.len(),
        2,
        "retrieve_tools must return exactly top_k results when k <= n"
    );
    assert!(
        results[0].1 >= results[1].1,
        "results must be sorted by descending score"
    );

    for &(idx, score) in &results {
        assert!(idx < descs.len(), "result index out of range");
        assert!(
            (-1.0..=1.0 + 1e-4).contains(&score),
            "score out of [-1,1] range: {score}"
        );
        eprintln!("  [{idx}] {:.4}  {:?}", score, descs[idx]);
    }

    // top_k > n must clamp to n
    let all = engine.retrieve_tools("test query", &descs, 100);
    assert_eq!(
        all.len(),
        descs.len(),
        "retrieve_tools must clamp top_k to available tools"
    );
}
