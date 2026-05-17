//! WASM bindings for Needle.
//!
//! Build: `wasm-pack build --target web crates/needle-wasm`
//! Or:    `wasm-pack build --target nodejs crates/needle-wasm`
//!
//! JS API:
//!   import init, { NeedleWasm } from './needle_wasm.js';
//!   await init();
//!   const engine = NeedleWasm.load(weightsUint8Array, vocabString);
//!   // single inference:
//!   const result = engine.run("What's the weather?", '[{"name":"get_weather",...}]');
//!   // streaming (fires per token):
//!   engine.run_stream(query, tools, (tokenId, piece) => process.stdout.write(piece));
//!   // batch:
//!   const results = engine.run_batch([{query:"...",tools:"..."}, ...]);
//!   // contrastive retrieval (requires weights with contrastive_proj_kernel):
//!   const emb = engine.encode_contrastive("What's the weather?"); // Float32Array | null

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use needle_infer::NeedleEngine;

/// WASM-exposed inference engine handle.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct NeedleWasm {
    engine: NeedleEngine,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl NeedleWasm {
    /// Load the engine from raw bytes.
    ///
    /// `weights_bytes`: ArrayBuffer / Uint8Array containing the .safetensors file.
    /// `vocab_text`:    String content of vocab.txt (one piece per line).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = load))]
    pub fn load(weights_bytes: Vec<u8>, vocab_text: String) -> Option<NeedleWasm> {
        match NeedleEngine::from_bytes(weights_bytes, &vocab_text) {
            Ok(engine) => Some(NeedleWasm { engine }),
            Err(e) => {
                #[cfg(target_arch = "wasm32")]
                web_sys_log(&format!("[needle-wasm] load error: {e}"));
                #[cfg(not(target_arch = "wasm32"))]
                eprintln!("[needle-wasm] load error: {e}");
                None
            }
        }
    }

    /// Run inference and return the output JSON string.
    ///
    /// Returns e.g. `[{"name":"get_weather","arguments":{"location":"Paris"}}]`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = run))]
    pub fn run(&self, query: &str, tools_json: &str) -> String {
        self.engine.run(query, tools_json).text
    }

    /// Encode text to a L2-normalized contrastive embedding.
    ///
    /// Returns a `Float32Array` of length `contrastive_dim()`, or `null` if the
    /// model was loaded without a contrastive head.
    ///
    /// Cosine similarity between query and tool embeddings equals the dot product
    /// (both vectors are already L2-normalized).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = encode_contrastive))]
    pub fn encode_contrastive(&self, text: &str) -> Option<Vec<f32>> {
        self.engine.encode_contrastive(text)
    }

    /// Return the contrastive embedding dimension (0 if no contrastive head loaded).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = contrastive_dim))]
    pub fn contrastive_dim(&self) -> usize {
        self.engine.contrastive_dim()
    }

    /// Rank tool descriptions by contrastive similarity to a query.
    ///
    /// `tool_descs_json`: JSON array of description strings, e.g.
    ///   `'["Get current weather", "Search the web", "Send email"]'`
    ///
    /// Returns a JSON string `[{"index":0,"score":0.95},{"index":2,"score":0.71}]`
    /// sorted by descending score, or `"[]"` if the model has no contrastive head.
    ///
    /// Mirrors Python `retrieve_tools(query, tools, top_k)` from `run.py`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = retrieve_tools))]
    pub fn retrieve_tools(&self, query: &str, tool_descs_json: &str, top_k: usize) -> String {
        let descs: Vec<String> = match serde_json::from_str(tool_descs_json) {
            Ok(v) => v,
            Err(_) => return "[]".to_string(),
        };
        let desc_refs: Vec<&str> = descs.iter().map(|s| s.as_str()).collect();
        let results = self.engine.retrieve_tools(query, &desc_refs, top_k);
        // Serialize as [{index, score}, ...]
        let mut out = String::from("[");
        for (i, (idx, score)) in results.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(r#"{{"index":{},"score":{:.6}}}"#, idx, score));
        }
        out.push(']');
        out
    }
}

// ── WASM-only methods that use JS types ──────────────────────────────────────
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl NeedleWasm {
    /// Run inference with a per-token streaming callback.
    ///
    /// `on_token(tokenId: number, piece: string)` fires for each generated token
    /// in decode order. Returns the final post-processed output string.
    #[wasm_bindgen(js_name = run_stream)]
    pub fn run_stream(&self, query: &str, tools_json: &str, on_token: &js_sys::Function) -> String {
        let this = JsValue::null();
        self.engine
            .run_stream(query, tools_json, |token_id, piece| {
                let _ = on_token.call2(
                    &this,
                    &JsValue::from_f64(token_id as f64),
                    &JsValue::from_str(piece),
                );
            })
            .text
    }

    /// Run inference on multiple examples and return a JS Array of output strings.
    ///
    /// Input: JS Array of `{query: string, tools: string}` objects.
    /// Output: JS Array of output strings, one per input.
    #[wasm_bindgen(js_name = run_batch)]
    pub fn run_batch(&self, examples: &js_sys::Array) -> js_sys::Array {
        let out = js_sys::Array::new();
        for i in 0..examples.length() {
            let ex = examples.get(i);
            let query = js_sys::Reflect::get(&ex, &JsValue::from_str("query"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            let tools = js_sys::Reflect::get(&ex, &JsValue::from_str("tools"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            out.push(&JsValue::from_str(&self.engine.run(&query, &tools).text));
        }
        out
    }
}

#[cfg(target_arch = "wasm32")]
fn web_sys_log(msg: &str) {
    web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(msg));
}
