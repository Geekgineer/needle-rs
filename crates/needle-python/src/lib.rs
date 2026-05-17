use needle_infer::engine::NeedleEngine;
use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;

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
    fn run_stream(&self, py: Python, query: &str, tools_json: &str, callback: PyObject) -> String {
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

#[pymodule]
fn needle_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNeedleEngine>()?;
    Ok(())
}
