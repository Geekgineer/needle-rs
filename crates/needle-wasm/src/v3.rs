//! Needle 3 in the browser.
//!
//! Mirrors [`crate::v2::NeedleV2Wasm`] so a page can switch generations by
//! swapping the class, with two deliberate differences:
//!
//! * There is no retrieval. v3 exports a confidence head and nothing else, so
//!   `contrastive_dim()` would always be 0 — the method is absent rather than
//!   present-and-useless.
//! * `confidence_for` matters more here than it did on v2. See its note.

use needle_infer::v3_engine::{extract_tool_call, KvPrecision, V3Engine, V3Options};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// A loaded Needle 3 model.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct NeedleV3Wasm {
    engine: V3Engine,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl NeedleV3Wasm {
    /// Load from the bytes of a `needle3.cact` container.
    ///
    /// Returns `undefined` if the bytes are not a v3 container — including
    /// when they are a *v2* container, which has its own class.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = load))]
    pub fn load(cact_bytes: Vec<u8>) -> Option<NeedleV3Wasm> {
        V3Engine::from_bytes(cact_bytes)
            .ok()
            .map(|engine| NeedleV3Wasm { engine })
    }

    /// The full completion, reasoning included.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = run))]
    pub fn run(&self, query: &str, tools_json: &str) -> String {
        self.engine.run(query, tools_json)
    }

    /// Just the tool-call payload.
    ///
    /// `"[]"` is a deliberate abstention — the model considered the tools and
    /// declined — and `""` means no `<tool_call>` markers were emitted at all.
    /// Treating the first as a failure turns a considered "no" into an error.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = run_json))]
    pub fn run_json(&self, query: &str, tools_json: &str) -> String {
        self.engine.run_json(query, tools_json).unwrap_or_default()
    }

    /// The model's chain-of-thought, if it produced any.
    ///
    /// v3 reasons on essentially every query; v2 only sometimes, and v1 never.
    /// A `<think>` block is not a reliable marker of which generation ran.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = reasoning))]
    pub fn reasoning(&self, text: &str) -> Option<String> {
        V3Engine::reasoning(text).map(str::to_string)
    }

    /// Generate with explicit settings.
    ///
    /// `kv_int8` stores the key/value cache as 8-bit — the width the container
    /// declares and upstream's engine runs — for roughly a quarter of the
    /// memory. See [`Self::kv_bytes`].
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = generate))]
    // The JS signature mirrors `V3Options` field for field; collapsing it into
    // an options object would mean hand-writing the glue wasm-bindgen gives us.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        query: &str,
        tools_json: &str,
        max_new_tokens: usize,
        temperature: f32,
        seed: u32,
        constrain: bool,
        kv_int8: bool,
    ) -> String {
        let opts = V3Options {
            max_new_tokens: if max_new_tokens == 0 {
                needle_infer::v3_engine::DEFAULT_MAX_NEW_TOKENS
            } else {
                max_new_tokens
            },
            temperature,
            seed: seed as u64,
            system: None,
            constrain,
            // A browser tab is exactly the constrained target this is for: a
            // full-context session costs 42.3 MB of cache at f32 and 11.5 MB
            // here, measured by `v3_kv_int8`.
            kv_precision: if kv_int8 {
                KvPrecision::Int8
            } else {
                KvPrecision::F32
            },
        };
        self.engine.generate(query, tools_json, &opts).text
    }

    /// How confident the model is in a completion it produced.
    ///
    /// **Pass the completion, not the query.** The head scores a finished
    /// judgement. On the shipped checkpoint a correct call scores 0.93, a
    /// wrong one 0.26 — and a bare query scores 0.80, which looks like a
    /// confident answer and is not one. v2 collapsed to near zero on a bare
    /// query, so the misuse announced itself; v3 does not.
    ///
    /// Returns `undefined` if the container exports no confidence head.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = confidence_for))]
    pub fn confidence_for(&self, query: &str, tools_json: &str, completion: &str) -> Option<f32> {
        self.engine.confidence_for(query, tools_json, completion)
    }

    /// Whether this container carries a confidence head.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = has_confidence))]
    pub fn has_confidence(&self) -> bool {
        self.engine.confidence.is_some()
    }

    /// Working-set estimate in bytes for a session of `seq_len` positions.
    ///
    /// Browser tabs have a memory budget and v3's container is 35 MB, so this
    /// is exposed rather than left for a page to guess. Counts the key/value
    /// cache only; the packed weights are a separate, fixed cost.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = kv_bytes))]
    pub fn kv_bytes(&self, seq_len: usize) -> usize {
        self.engine.model.cfg.kv_bytes(seq_len, 4)
    }

    /// The same estimate for a cache stored at int8.
    ///
    /// Not a flat quarter of [`Self::kv_bytes`]: every stored head vector also
    /// carries an `f32` scale. At full context it is 27% of the f32 figure.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = kv_bytes_int8))]
    pub fn kv_bytes_int8(&self, seq_len: usize) -> usize {
        self.engine
            .model
            .cfg
            .kv_bytes_at(seq_len, KvPrecision::Int8)
    }

    /// Context limit in tokens.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = max_seq_len))]
    pub fn max_seq_len(&self) -> usize {
        self.engine.model.cfg.max_seq_len
    }
}

/// Streaming, which needs a JS callback and so is wasm-only.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl NeedleV3Wasm {
    /// Generate, invoking `on_token(text)` with each decoded delta.
    ///
    /// The callback receives decoded text, not raw pieces — a newline arrives
    /// as the byte-fallback token `<0x0A>`, and appending raw pieces would
    /// render that literally. Concatenating every delta reproduces the return
    /// value exactly.
    #[wasm_bindgen(js_name = run_stream)]
    pub fn run_stream(&self, query: &str, tools_json: &str, on_token: &js_sys::Function) -> String {
        let this = JsValue::NULL;
        self.engine
            .generate_with(query, tools_json, &V3Options::default(), |_id, piece| {
                let _ = on_token.call1(&this, &JsValue::from_str(piece));
            })
            .text
    }
}

/// Pull the payload out of a completion the caller already has.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = extract_tool_call_v3))]
pub fn extract_tool_call_v3(text: &str) -> Option<String> {
    extract_tool_call(text)
}
