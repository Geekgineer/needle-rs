use needle_infer::engine::NeedleEngine;
use needle_infer::v2_engine::{GenerateOptions, V2Engine};
use needle_infer::v3_engine::{
    extract_tool_call, KvPrecision, V3Engine, V3Options, DEFAULT_MAX_NEW_TOKENS,
};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

#[pyclass(name = "NeedleEngine", module = "needle_rs")]
struct PyNeedleEngine {
    inner: NeedleEngine,
}

#[pymethods]
impl PyNeedleEngine {
    /// Load from file paths.
    #[staticmethod]
    fn load(weights_path: &str, vocab_path: &str) -> PyResult<Self> {
        NeedleEngine::load(weights_path, vocab_path)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load from in-memory bytes (useful when the caller already has the weights buffered).
    #[staticmethod]
    fn from_bytes(weights_bytes: &[u8], vocab_text: &str) -> PyResult<Self> {
        NeedleEngine::from_bytes(weights_bytes.to_vec(), vocab_text)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Run inference and return the final JSON tool-call string.
    fn run(&self, query: &str, tools_json: &str) -> String {
        self.inner.run(query, tools_json).text
    }

    /// Run inference with a per-token callback, return the final JSON string.
    ///
    /// `callback` is called as `callback(token_id: int, piece: str)` for each
    /// generated token piece. The return value is the same post-processed string
    /// as `run()`.
    fn run_stream(&self, py: Python, query: &str, tools_json: &str, callback: Py<PyAny>) -> String {
        self.inner
            .run_stream(query, tools_json, |token_id, piece| {
                let _ = callback.call1(py, (token_id, piece));
            })
            .text
    }

    /// Run inference on a batch of (query, tools_json) pairs.
    ///
    /// Returns a list of result strings in the same order as the input.
    fn run_batch(&self, examples: Vec<(String, String)>) -> Vec<String> {
        let pairs: Vec<(&str, &str)> = examples
            .iter()
            .map(|(q, t)| (q.as_str(), t.as_str()))
            .collect();
        self.inner
            .run_batch(&pairs)
            .into_iter()
            .map(|r| r.text)
            .collect()
    }

    /// Return the L2-normalised contrastive embedding for `text`, or None if
    /// the loaded weights have no contrastive head.
    fn encode_contrastive(&self, text: &str) -> Option<Vec<f32>> {
        self.inner.encode_contrastive(text)
    }

    /// Dimension of the contrastive embedding (0 if no contrastive head).
    fn contrastive_dim(&self) -> usize {
        self.inner.contrastive_dim()
    }

    /// Rank `tool_descriptions` by cosine similarity to `query`.
    ///
    /// Returns a list of `(index, score)` tuples sorted by descending score,
    /// truncated to `top_k`. Returns an empty list if no contrastive head is
    /// present in the loaded weights.
    fn retrieve_tools(
        &self,
        query: &str,
        tool_descriptions: Vec<String>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let descs: Vec<&str> = tool_descriptions.iter().map(|s| s.as_str()).collect();
        self.inner.retrieve_tools(query, &descs, top_k)
    }
}

/// Needle v2. A `.cact` container carries its weights, geometry and tokenizer,
/// so `load` takes one path and there is no vocabulary argument.
#[pyclass(name = "V2Engine", module = "needle_rs")]
struct PyV2Engine {
    inner: V2Engine,
}

#[pymethods]
impl PyV2Engine {
    /// Load a `.cact` model from disk.
    #[staticmethod]
    fn load(cact_path: &str) -> PyResult<Self> {
        V2Engine::load(cact_path)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load a `.cact` model from an in-memory image.
    #[staticmethod]
    fn from_bytes(cact_bytes: &[u8]) -> PyResult<Self> {
        V2Engine::from_bytes(cact_bytes.to_vec())
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Greedy tool call. Returns the full decoded text.
    fn run(&self, query: &str, tools_json: &str) -> String {
        self.inner.run(query, tools_json).text
    }

    /// Just the `<tool_call>` payload, or None if the model emitted no call.
    fn run_json(&self, query: &str, tools_json: &str) -> Option<String> {
        self.inner.run(query, tools_json).tool_call
    }

    /// Generation with explicit settings.
    ///
    /// Returns a dict: `text`, `tool_call`, `thinking`, `token_ids`,
    /// `prompt_tokens`, `stop_reason`.
    #[pyo3(signature = (query, tools_json, max_new_tokens=128, temperature=0.0, seed=0,
                        system=None, constrain=false))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        tools_json: &str,
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
        system: Option<String>,
        constrain: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let opts = GenerateOptions {
            max_new_tokens,
            temperature,
            seed,
            system,
            constrain,
            ..Default::default()
        };
        let r = self.inner.generate(query, tools_json, &opts, |_, _| {});
        // Read the derived fields before the struct is taken apart.
        let stop_reason = format!("{:?}", r.stop_reason);
        let stopped_naturally = r.stopped_naturally();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("text", r.text)?;
        d.set_item("tool_call", r.tool_call)?;
        d.set_item("thinking", r.thinking)?;
        d.set_item("token_ids", r.token_ids)?;
        d.set_item("prompt_tokens", r.prompt_tokens)?;
        d.set_item("stop_reason", stop_reason)?;
        d.set_item("stopped_naturally", stopped_naturally)?;
        Ok(d)
    }

    /// Streaming generation. `callback(token_id: int, piece: str)` per token.
    fn run_stream(&self, py: Python, query: &str, tools_json: &str, callback: Py<PyAny>) -> String {
        self.inner
            .run_stream(query, tools_json, |token_id, piece| {
                let _ = callback.call1(py, (token_id, piece));
            })
            .text
    }

    /// Run several queries in sequence, reusing one KV cache.
    fn run_batch(&self, examples: Vec<(String, String)>) -> Vec<String> {
        let pairs: Vec<(&str, &str)> = examples
            .iter()
            .map(|(q, t)| (q.as_str(), t.as_str()))
            .collect();
        self.inner
            .run_batch(&pairs)
            .into_iter()
            .map(|r| r.text)
            .collect()
    }

    /// L2-normalised contrastive embedding, or None without such a head.
    fn encode_contrastive(&self, text: &str) -> Option<Vec<f32>> {
        self.inner.encode_contrastive(text)
    }

    /// Width of the contrastive embedding (0 if absent).
    fn contrastive_dim(&self) -> usize {
        self.inner.contrastive_dim()
    }

    /// Probability in (0, 1) that `completion` is the right answer for this
    /// query, or None without a confidence head.
    ///
    /// The confidence head scores a judgement already made, so it needs the
    /// completion: pass what `run()` returned. Gate on it — act above a
    /// threshold you pick, escalate below it.
    fn confidence_for(&self, query: &str, tools_json: &str, completion: &str) -> Option<f32> {
        self.inner.confidence_for(query, tools_json, completion)
    }

    /// Raw confidence logit for an arbitrary string, or None without the head.
    ///
    /// The primitive. A bare query reads near zero however answerable it is,
    /// because the completion is what gets scored — prefer `confidence_for`.
    fn confidence(&self, text: &str) -> Option<f32> {
        self.inner.confidence(text)
    }

    /// `confidence()` as a probability in (0, 1).
    fn confidence_probability(&self, text: &str) -> Option<f32> {
        self.inner.confidence_probability(text)
    }

    /// Rank tool descriptions by cosine similarity to `query`.
    fn retrieve_tools(
        &self,
        query: &str,
        tool_descriptions: Vec<String>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let descs: Vec<&str> = tool_descriptions.iter().map(String::as_str).collect();
        self.inner.retrieve_tools(query, &descs, top_k)
    }

    /// Model geometry, as a dict.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let c = &self.inner.model().cfg;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("vocab_size", c.vocab_size)?;
        d.set_item("d_model", c.d_model)?;
        d.set_item("num_heads", c.num_heads)?;
        d.set_item("num_kv_heads", c.num_kv_heads)?;
        d.set_item("num_layers", c.num_layers)?;
        d.set_item("head_dim", c.head_dim)?;
        d.set_item("max_seq_len", c.max_seq_len)?;
        d.set_item("mhc_lanes", c.mhc_lanes)?;
        d.set_item("kv_window", c.kv_window)?;
        d.set_item("rope_theta", c.rope_theta)?;
        d.set_item("engram_sites", c.engram.sites.clone())?;
        Ok(d)
    }
}

/// Needle 3.
///
/// Mirrors `V2Engine` so switching generations is a class swap, with two
/// differences that follow from the model rather than from taste:
///
/// * No `encode_contrastive` or `retrieve_tools`. v3 exports a confidence head
///   and nothing else, so they would return empty on every call.
/// * `reasoning()`, because v3 emits a `<think>` block on essentially every
///   query. v2 does so only sometimes, so the block is not a generation marker.
#[pyclass(name = "V3Engine", module = "needle_rs")]
struct PyV3Engine {
    inner: V3Engine,
}

#[pymethods]
impl PyV3Engine {
    /// Load a `needle3.cact` model from disk.
    #[staticmethod]
    fn load(cact_path: &str) -> PyResult<Self> {
        V3Engine::load(cact_path)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load a `needle3.cact` model from an in-memory image.
    #[staticmethod]
    fn from_bytes(cact_bytes: &[u8]) -> PyResult<Self> {
        V3Engine::from_bytes(cact_bytes.to_vec())
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// The full completion, reasoning included.
    fn run(&self, query: &str, tools_json: &str) -> String {
        self.inner.run(query, tools_json)
    }

    /// Just the tool-call payload.
    ///
    /// `"[]"` is a deliberate abstention — the model weighed the tools and
    /// declined — where `None` means it emitted no `<tool_call>` markers at
    /// all. Treating the first as a failure turns a considered "no" into an
    /// error.
    fn run_json(&self, query: &str, tools_json: &str) -> Option<String> {
        self.inner.run_json(query, tools_json)
    }

    /// The chain-of-thought inside a completion, if it has one.
    #[staticmethod]
    fn reasoning(text: &str) -> Option<String> {
        V3Engine::reasoning(text).map(str::to_string)
    }

    /// Pull the payload out of a completion you already have.
    #[staticmethod]
    fn extract_tool_call(text: &str) -> Option<String> {
        extract_tool_call(text)
    }

    /// Generation with explicit settings.
    ///
    /// Returns a dict: `text`, `tool_call`, `reasoning`, `token_ids`,
    /// `stop_reason`, `positions`.
    #[pyo3(signature = (query, tools_json, max_new_tokens=0, temperature=0.0, seed=0,
                        system=None, constrain=false, kv_int8=false))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        tools_json: &str,
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
        system: Option<String>,
        constrain: bool,
        kv_int8: bool,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let opts = V3Options {
            max_new_tokens: if max_new_tokens == 0 {
                DEFAULT_MAX_NEW_TOKENS
            } else {
                max_new_tokens
            },
            temperature,
            seed,
            system,
            constrain,
            kv_precision: if kv_int8 {
                KvPrecision::Int8
            } else {
                KvPrecision::F32
            },
        };
        let r = self.inner.generate(query, tools_json, &opts);
        let d = pyo3::types::PyDict::new(py);
        d.set_item("text", &r.text)?;
        d.set_item("tool_call", extract_tool_call(&r.text))?;
        d.set_item("reasoning", V3Engine::reasoning(&r.text))?;
        d.set_item("token_ids", r.tokens.clone())?;
        d.set_item("stop_reason", format!("{:?}", r.stop))?;
        d.set_item("positions", r.positions)?;
        Ok(d)
    }

    /// Generate, calling `callback(piece)` with each decoded delta.
    ///
    /// The callback receives decoded text, not raw tokenizer pieces:
    /// concatenating every delta reproduces the return value exactly.
    fn run_stream(&self, py: Python, query: &str, tools_json: &str, callback: Py<PyAny>) -> String {
        self.inner
            .generate_with(query, tools_json, &V3Options::default(), |_id, piece| {
                let _ = callback.call1(py, (piece,));
            })
            .text
    }

    /// Whether this container carries a confidence head.
    fn has_confidence(&self) -> bool {
        self.inner.confidence.is_some()
    }

    /// How confident the model is in a completion it produced.
    ///
    /// Pass the **completion**, not the query. The head scores a finished
    /// judgement. On the shipped checkpoint a correct call scores 0.93 and a
    /// wrong one 0.26 — but a bare query scores 0.80, which looks like a
    /// confident answer and is not one. v2 collapsed to near zero on a bare
    /// query, so that misuse announced itself; v3's does not.
    fn confidence_for(&self, query: &str, tools_json: &str, completion: &str) -> Option<f32> {
        self.inner.confidence_for(query, tools_json, completion)
    }

    /// Generate, then score what was generated. Returns `(result, confidence)`.
    fn run_scored<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        tools_json: &str,
    ) -> PyResult<(Bound<'py, pyo3::types::PyDict>, Option<f32>)> {
        let (r, p) = self.inner.run_scored(query, tools_json);
        let d = pyo3::types::PyDict::new(py);
        d.set_item("text", &r.text)?;
        d.set_item("tool_call", extract_tool_call(&r.text))?;
        d.set_item("reasoning", V3Engine::reasoning(&r.text))?;
        d.set_item("stop_reason", format!("{:?}", r.stop))?;
        Ok((d, p))
    }

    /// Key/value cache bytes for a session of `seq_len` positions.
    ///
    /// Sized to the session rather than to the context limit, so this is what
    /// a run actually costs — useful when deciding whether one fits. Pass
    /// `kv_int8=True` to get the figure for the cache `generate(kv_int8=True)`
    /// allocates; it is 27% of the f32 figure rather than 25%, because each
    /// stored head vector also carries a scale.
    #[pyo3(signature = (seq_len, kv_int8=false))]
    fn kv_bytes(&self, seq_len: usize, kv_int8: bool) -> usize {
        let p = if kv_int8 {
            KvPrecision::Int8
        } else {
            KvPrecision::F32
        };
        self.inner.model.cfg.kv_bytes_at(seq_len, p)
    }

    /// Context limit in tokens.
    #[getter]
    fn max_seq_len(&self) -> usize {
        self.inner.model.cfg.max_seq_len
    }
}

#[pymodule]
fn needle_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNeedleEngine>()?;
    m.add_class::<PyV2Engine>()?;
    m.add_class::<PyV3Engine>()?;
    Ok(())
}
