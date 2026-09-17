//! High-level inference for Needle v2: prompt assembly, generation, streaming.
//!
//! Everything a caller needs travels inside the `.cact` file — weights, geometry,
//! and the tokenizer — so a session is one path and no side files.
//!
//! ```no_run
//! use needle_infer::v2_engine::V2Engine;
//! let engine = V2Engine::load("weights/needle2.cact")?;
//! let out = engine.run("What's the weather in Paris?", r#"[{"name":"get_weather"}]"#);
//! println!("{}", out.text);
//! # Ok::<(), std::io::Error>(())
//! ```

use crate::constrained::{ConstrainedDecoder, ToolDef};
use crate::sp_tokenizer::SpTokenizer;
use crate::v2::{V2Bundle, V2LoadError};
use needle_core::v2::{
    V2Batch, V2Model, V2State, DEFAULT_CHUNK, HEAD_CONFIDENCE, HEAD_CONTRASTIVE,
};
use std::path::Path;

pub use crate::prompt::{
    build_prompt as build_chat_prompt, compact_json, IM_END, IM_START, THINK_END, THINK_START,
    TOOLS_END, TOOLS_START, TOOL_CALL_END, TOOL_CALL_START,
};

/// Default generation cap.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 128;

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted the end-of-sequence token.
    Eos,
    /// The model closed the assistant turn with `<|im_end|>`.
    ImEnd,
    /// `max_new_tokens` was reached with the turn still open.
    MaxTokens,
    /// `max_seq_len` was reached.
    ContextFull,
    /// The forward pass rejected a step. Distinct from a cap: the output is not a
    /// completed turn that ran long, it is a run that could not continue.
    Error(&'static str),
}

impl StopReason {
    /// True when the model chose to stop, rather than being cut off.
    pub fn is_natural(self) -> bool {
        matches!(self, Self::Eos | Self::ImEnd)
    }
}

/// One completion.
#[derive(Debug, Clone)]
pub struct V2Result {
    /// Decoded text, markers included.
    pub text: String,
    /// The tool-call payload between `<tool_call>` and `</tool_call>`, when the
    /// model emitted one.
    pub tool_call: Option<String>,
    /// Reasoning between `<think>` and `</think>`, when present.
    pub thinking: Option<String>,
    pub token_ids: Vec<u32>,
    /// Prompt tokens fed, including BOS.
    pub prompt_tokens: usize,
    pub stop_reason: StopReason,
}

impl V2Result {
    /// True when the model chose to stop rather than being cut off.
    pub fn stopped_naturally(&self) -> bool {
        self.stop_reason.is_natural()
    }

    /// The error that ended the run, if any.
    pub fn error(&self) -> Option<&'static str> {
        match self.stop_reason {
            StopReason::Error(e) => Some(e),
            _ => None,
        }
    }

    fn failed(prompt_tokens: usize, e: &'static str) -> Self {
        Self {
            text: String::new(),
            tool_call: None,
            thinking: None,
            token_ids: Vec::new(),
            prompt_tokens,
            stop_reason: StopReason::Error(e),
        }
    }
}

/// Generation settings.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    /// `0.0` is greedy. Above that, temperature sampling with `seed`.
    pub temperature: f32,
    pub seed: u64,
    /// Optional system message, placed before the user turn.
    pub system: Option<String>,
    /// Positions per batched-prefill chunk. `0` disables batching and prefills
    /// one position at a time via the reference path.
    pub prefill_chunk: usize,
    /// Constrain the tool-call payload to the declared tool names and argument
    /// keys. Applies only between `<tool_call>` and `</tool_call>`; the rest of
    /// the turn, including `<think>`, is left free.
    pub constrain: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            temperature: 0.0,
            seed: 0,
            system: None,
            prefill_chunk: DEFAULT_CHUNK,
            constrain: false,
        }
    }
}

/// A loaded v2 model plus its tokenizer.
pub struct V2Engine {
    bundle: V2Bundle,
}

impl V2Engine {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Ok(Self {
            bundle: V2Bundle::load(path)?,
        })
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, V2LoadError> {
        Ok(Self {
            bundle: V2Bundle::from_bytes(bytes)?,
        })
    }

    pub fn model(&self) -> &V2Model {
        &self.bundle.model
    }

    pub fn bundle(&self) -> &V2Bundle {
        &self.bundle
    }

    /// The embedded tokenizer. Absent only for a container exported without one.
    pub fn tokenizer(&self) -> Option<&SpTokenizer> {
        self.bundle.tokenizer.as_ref()
    }

    fn tok(&self) -> &SpTokenizer {
        self.bundle
            .tokenizer
            .as_ref()
            .expect("container has no embedded tokenizer; use the lower-level V2Bundle API")
    }

    /// Assemble the v2 chat prompt.
    ///
    /// ```text
    /// [<|im_start|>system\n{system}<|im_end|>\n]
    /// <|im_start|>user\n<tools>{tools}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n
    /// ```
    ///
    /// `tools_json` is compacted before embedding. This matters: the model was
    /// trained on compact schemas, and the indentation a caller naturally gets
    /// from `JSON.stringify(x, null, 2)` or `json.dumps(x, indent=2)` is enough to
    /// change the decision. On the shipped checkpoint, the same query and schema
    /// yields `[{"name":"get_weather",...}]` compact and `[]` pretty-printed.
    /// Rather than make that a documented gotcha, the whitespace is removed here.
    pub fn build_prompt(query: &str, tools_json: &str, system: Option<&str>) -> String {
        crate::prompt::build_prompt(query, tools_json, system)
    }

    /// Greedy tool call for one query.
    pub fn run(&self, query: &str, tools_json: &str) -> V2Result {
        self.generate(query, tools_json, &GenerateOptions::default(), |_, _| {})
    }

    /// Like [`run`], but `on_token(id, piece)` fires per generated token.
    ///
    /// [`run`]: V2Engine::run
    pub fn run_stream<F: FnMut(u32, &str)>(
        &self,
        query: &str,
        tools_json: &str,
        on_token: F,
    ) -> V2Result {
        self.generate(query, tools_json, &GenerateOptions::default(), on_token)
    }

    /// Full-control generation, allocating a fresh sequence state.
    ///
    /// A `V2State` owns the KV cache, which at the shipped geometry is tens of
    /// megabytes. Use [`new_state`] plus [`generate_with_state`] to reuse one
    /// across queries.
    ///
    /// [`new_state`]: V2Engine::new_state
    /// [`generate_with_state`]: V2Engine::generate_with_state
    pub fn generate<F: FnMut(u32, &str)>(
        &self,
        query: &str,
        tools_json: &str,
        opts: &GenerateOptions,
        on_token: F,
    ) -> V2Result {
        let mut state = self.new_state();
        self.generate_with_state(query, tools_json, opts, &mut state, on_token)
    }

    /// Allocate a reusable sequence state.
    pub fn new_state(&self) -> V2State {
        self.bundle.model.make_state()
    }

    /// Generation with the prompt prefilled one position at a time.
    ///
    /// Same result as [`generate`], just without batching. Kept public because it
    /// is the reference the batched path is tested against, and because a caller
    /// that is memory-constrained may prefer it: it needs no [`V2Batch`] scratch.
    ///
    /// [`generate`]: V2Engine::generate
    pub fn generate_sequential<F: FnMut(u32, &str)>(
        &self,
        query: &str,
        tools_json: &str,
        opts: &GenerateOptions,
        on_token: F,
    ) -> V2Result {
        let mut state = self.new_state();
        let mut o = opts.clone();
        o.prefill_chunk = 0;
        self.generate_with_state(query, tools_json, &o, &mut state, on_token)
    }

    /// Generate into a caller-owned state, which is reset first.
    ///
    /// Reusing a state across queries avoids reallocating the KV cache each time.
    pub fn generate_with_state<F: FnMut(u32, &str)>(
        &self,
        query: &str,
        tools_json: &str,
        opts: &GenerateOptions,
        state: &mut V2State,
        mut on_token: F,
    ) -> V2Result {
        let model = &self.bundle.model;
        let tok = self.tok();
        let prompt = Self::build_prompt(query, tools_json, opts.system.as_deref());

        // BOS then the prompt, truncated so the cap still fits inside max_seq_len.
        let mut ids = Vec::with_capacity(64);
        ids.push(tok.bos_id);
        ids.extend(tok.encode(&prompt));
        let room = model
            .cfg
            .max_seq_len
            .saturating_sub(opts.max_new_tokens)
            .max(1);
        if ids.len() > room {
            ids.truncate(room);
        }
        let prompt_tokens = ids.len();

        state.reset();
        let mut logits = vec![0.0f32; model.cfg.vocab_size];
        // Only the final prompt position's logits are sampled from, so the rest
        // skip the tied LM head — the widest matmul in a step.
        if opts.prefill_chunk > 0 {
            let mut batch = V2Batch::new(model, opts.prefill_chunk);
            if let Err(e) = model.prefill_batch(&ids, state, &mut batch, Some(&mut logits)) {
                return V2Result::failed(prompt_tokens, e);
            }
        } else {
            let (last_prompt, prefix) = ids.split_last().expect("prompt has at least BOS");
            for &t in prefix {
                if let Err(e) = model.step_prefill(t, state) {
                    return V2Result::failed(prompt_tokens, e);
                }
            }
            if let Err(e) = model.step(*last_prompt, state, &mut logits) {
                return V2Result::failed(prompt_tokens, e);
            }
        }

        let im_end = tok.id_of(IM_END);
        let mut rng = SplitMix64::new(opts.seed);
        let mut out_ids: Vec<u32> = Vec::new();
        let mut stop = StopReason::MaxTokens;

        // Grammar constraint, engaged only inside the tool-call payload. Running
        // it over the whole turn would risk a `"name":"` inside `<think>` prose
        // putting the state machine into a constrained state where it does not
        // belong. Both markers are single user-defined tokens, so entering and
        // leaving is an id comparison rather than a text scan.
        let (tc_start, tc_end) = (tok.id_of(TOOL_CALL_START), tok.id_of(TOOL_CALL_END));
        let mut grammar = if opts.constrain {
            let defs = ToolDef::from_json(tools_json);
            // v2 additionally forbids repeating an argument key within a call.
            (!defs.is_empty())
                .then(|| ConstrainedDecoder::new(&defs, byte_table(tok)).with_unique_arg_keys())
        } else {
            None
        };
        let mut in_tool_call = false;
        let mut mask: Vec<f32> = Vec::new();

        for _ in 0..opts.max_new_tokens {
            if in_tool_call {
                if let Some(g) = grammar.as_ref() {
                    mask.clear();
                    mask.extend_from_slice(&g.logit_mask(model.cfg.vocab_size));
                    for (l, &m) in logits.iter_mut().zip(mask.iter()) {
                        *l += m;
                    }
                }
            }
            let next = if opts.temperature > 0.0 {
                sample(&logits, opts.temperature, &mut rng)
            } else {
                argmax(&logits)
            };
            if next == tok.eos_id {
                stop = StopReason::Eos;
                break;
            }
            if Some(next) == im_end {
                stop = StopReason::ImEnd;
                break;
            }
            out_ids.push(next);
            if grammar.is_some() {
                if Some(next) == tc_start {
                    in_tool_call = true;
                } else if Some(next) == tc_end {
                    in_tool_call = false;
                } else if in_tool_call {
                    if let Some(g) = grammar.as_mut() {
                        g.update(next);
                    }
                }
            }
            // Decoding one id at a time can split a multi-byte piece, so stream
            // the delta of the decoded prefix rather than the piece surface.
            let piece = incremental_piece(tok, &out_ids);
            on_token(next, &piece);

            if state.pos() >= model.cfg.max_seq_len {
                stop = StopReason::ContextFull;
                break;
            }
            if let Err(e) = model.step(next, state, &mut logits) {
                stop = StopReason::Error(e);
                break;
            }
        }

        let text = tok.decode(&out_ids);
        V2Result {
            tool_call: extract_between(&text, TOOL_CALL_START, TOOL_CALL_END),
            thinking: extract_between(&text, THINK_START, THINK_END),
            text,
            token_ids: out_ids,
            prompt_tokens,
            stop_reason: stop,
        }
    }

    /// Run several queries in sequence, reusing one state so the KV cache is
    /// allocated once.
    ///
    /// Sequential, not a batching path: each query is a separate forward pass.
    pub fn run_batch(&self, examples: &[(&str, &str)]) -> Vec<V2Result> {
        let opts = GenerateOptions::default();
        let mut state = self.new_state();
        examples
            .iter()
            .map(|(q, t)| self.generate_with_state(q, t, &opts, &mut state, |_, _| {}))
            .collect()
    }

    /// Feed a prompt and return the post-final-norm hidden state of the last
    /// position.
    ///
    /// This is the vector the LM head consumes. It is **not** what the probe
    /// heads pool over — see [`encode_contrastive`].
    ///
    /// [`encode_contrastive`]: V2Engine::encode_contrastive
    pub fn last_hidden(&self, text: &str) -> Option<Vec<f32>> {
        let model = &self.bundle.model;
        let ids = self.encode_with_bos(text)?;
        let mut state = model.make_state();
        // No sampling here, so the LM head is never needed.
        for &t in &ids {
            model.step_prefill(t, &mut state).ok()?;
        }
        Some(V2Model::last_hidden(&state).to_vec())
    }

    fn encode_with_bos(&self, text: &str) -> Option<Vec<u32>> {
        let tok = self.bundle.tokenizer.as_ref()?;
        let mut ids = Vec::with_capacity(32);
        ids.push(tok.bos_id);
        ids.extend(tok.encode(text));
        ids.truncate(self.bundle.model.cfg.max_seq_len);
        Some(ids)
    }

    /// Run a probe head over `text`.
    ///
    /// The heads pool over **cells** — the scaled embedding plus the lane-mean of
    /// the residual stream after every layer, at every position — so this drives
    /// the forward pass with a trace callback and streams each cell into a
    /// [`ProbePool`] rather than retaining `T x (L+1) x d_model` floats.
    ///
    /// Two things differ from the LM path, both taken from
    /// `architecture.encode_contrastive`:
    ///
    /// * attention runs **full causal** (`window = 0`), not the checkpoint's
    ///   `kv_window`. Identical below 256 tokens, materially different above it.
    /// * cells at positions holding the pad token are dropped, matching
    ///   `make_padding_mask` and `probe_pool`'s `keep`.
    fn run_head(&self, code: u8, text: &str) -> Option<Vec<f32>> {
        let mut state = self.new_head_state()?;
        self.run_head_with_state(code, text, &mut state)
    }

    /// Allocate a state suitable for the probe heads.
    ///
    /// The heads attend over the whole sequence, so this is the full-length cache
    /// (113 MB at the shipped geometry) rather than the 14 MB ring generation
    /// uses. Hold one and pass it to [`encode_contrastive_with_state`] when
    /// embedding more than one string — [`retrieve_tools`] does.
    ///
    /// Returns `None` if the container carries no probe head.
    ///
    /// [`encode_contrastive_with_state`]: V2Engine::encode_contrastive_with_state
    /// [`retrieve_tools`]: V2Engine::retrieve_tools
    pub fn new_head_state(&self) -> Option<V2State> {
        if self.bundle.heads.is_empty() {
            return None;
        }
        Some(self.bundle.model.make_state_full_causal())
    }

    fn run_head_with_state(&self, code: u8, text: &str, state: &mut V2State) -> Option<Vec<f32>> {
        let model = &self.bundle.model;
        let head = self.bundle.head(code)?;
        let tok = self.bundle.tokenizer.as_ref()?;
        let ids = self.encode_with_bos(text)?;
        if ids.is_empty() {
            return None;
        }

        let mut pool = head.pool().ok()?;
        state.reset();
        // Full causal, not the checkpoint's window: `encode_contrastive` and
        // `forward_confidence` default to `window = 0`. This fails rather than
        // truncating if the state was not allocated for it.
        state.set_kv_window(Some(0), model).ok()?;

        let (d, n) = (model.cfg.d_model, model.cfg.mhc_lanes);
        let mut cell = vec![0.0f32; d];
        let mut logits = vec![0.0f32; model.cfg.vocab_size];

        for &t in &ids {
            let keep = t != tok.pad_id;
            {
                let pool = &mut pool;
                let cell = &mut cell;
                let mut trace = |key: &str, _layer: usize, values: &[f32]| {
                    if !keep {
                        return;
                    }
                    match key {
                        // cells[0]: the scaled embedding, before the lane broadcast.
                        "embed" => pool.push(values),
                        // cells[l+1]: the lane-mean after layer l.
                        "x" => {
                            for (c, slot) in cell.iter_mut().enumerate() {
                                let mut acc = 0.0f32;
                                for lane in 0..n {
                                    acc += values[lane * d + c];
                                }
                                *slot = acc / n as f32;
                            }
                            pool.push(cell);
                        }
                        _ => {}
                    }
                };
                model.step_traced(t, state, &mut logits, &mut trace).ok()?;
            }
        }

        let mut pooled = vec![0.0f32; pool.pooled_len()];
        pool.pooled(&mut pooled).ok()?;
        let mut out = vec![0.0f32; head.out_dim];
        if code == HEAD_CONTRASTIVE {
            head.project_normalized(&pooled, &mut out);
        } else {
            head.project(&pooled, &mut out);
        }
        Some(out)
    }

    /// L2-normalised contrastive embedding for `text`, or `None` if the
    /// container carries no contrastive head.
    ///
    /// Allocates a full-length cache per call; use
    /// [`encode_contrastive_with_state`] to embed several strings.
    ///
    /// [`encode_contrastive_with_state`]: V2Engine::encode_contrastive_with_state
    pub fn encode_contrastive(&self, text: &str) -> Option<Vec<f32>> {
        self.run_head(HEAD_CONTRASTIVE, text)
    }

    /// [`encode_contrastive`] into a caller-owned state, which is reset first.
    ///
    /// [`encode_contrastive`]: V2Engine::encode_contrastive
    pub fn encode_contrastive_with_state(
        &self,
        text: &str,
        state: &mut V2State,
    ) -> Option<Vec<f32>> {
        self.run_head_with_state(HEAD_CONTRASTIVE, text, state)
    }

    /// [`confidence`] into a caller-owned state, which is reset first.
    ///
    /// [`confidence`]: V2Engine::confidence
    pub fn confidence_with_state(&self, text: &str, state: &mut V2State) -> Option<f32> {
        self.run_head_with_state(HEAD_CONFIDENCE, text, state)
            .map(|v| v[0])
    }

    /// Width of the contrastive embedding, or 0 if there is no such head.
    pub fn contrastive_dim(&self) -> usize {
        self.bundle.contrastive_head().map_or(0, |h| h.out_dim)
    }

    /// Raw confidence logit for an arbitrary token sequence, or `None` if the
    /// container carries no confidence head. Higher is more confident.
    ///
    /// This is the primitive, equivalent to upstream `forward_confidence`. The
    /// head scores a *judgement already made*: it is trained on a formatted
    /// prompt followed by the completion, so scoring a bare query is
    /// meaningless — it reads near zero whatever the query. Prefer
    /// [`confidence_for`], which assembles the input correctly.
    ///
    /// [`confidence_for`]: V2Engine::confidence_for
    pub fn confidence(&self, text: &str) -> Option<f32> {
        self.run_head(HEAD_CONFIDENCE, text).map(|v| v[0])
    }

    /// [`confidence`] as a probability in `(0, 1)`.
    ///
    /// [`confidence`]: V2Engine::confidence
    pub fn confidence_probability(&self, text: &str) -> Option<f32> {
        self.confidence(text).map(|z| 1.0 / (1.0 + (-z).exp()))
    }

    /// Confidence that `completion` is the right answer for this query, as a
    /// probability in `(0, 1)`. `None` if the container has no confidence head.
    ///
    /// Feeds the head what it was trained on: the formatted prompt for
    /// `(query, tools_json)` followed by the model's own output. Pass the
    /// [`V2Result::text`] of the run being judged.
    ///
    /// ```no_run
    /// # use needle_infer::v2_engine::V2Engine;
    /// # let engine = V2Engine::load("weights/needle2.cact").unwrap();
    /// # let (query, tools) = ("What's the weather in Paris?", "[]");
    /// let result = engine.run(query, tools);
    /// if engine.confidence_for(query, tools, &result.text).unwrap_or(0.0) < 0.5 {
    ///     // escalate rather than execute
    /// }
    /// ```
    ///
    /// Upstream additionally takes the minimum of this head and the decode
    /// probability of the call tokens; that composition is not replicated here
    /// because the reference implementation is not published. Calibration also
    /// holds for the base model only — see `docs/hf-model-card.md`.
    pub fn confidence_for(&self, query: &str, tools_json: &str, completion: &str) -> Option<f32> {
        let mut text = Self::build_prompt(query, tools_json, None);
        text.push_str(completion);
        self.confidence(&text).map(|z| 1.0 / (1.0 + (-z).exp()))
    }

    /// Rank tool descriptions by cosine similarity to `query`.
    ///
    /// Returns `(index, score)` sorted by descending score, truncated to
    /// `top_k`. Empty when the container has no contrastive head.
    pub fn retrieve_tools(
        &self,
        query: &str,
        tool_descriptions: &[&str],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // One state for the query and every description: a head state is the
        // full-length cache, so allocating per description would mean tens of
        // megabytes churned per call.
        let Some(mut state) = self.new_head_state() else {
            return Vec::new();
        };
        let Some(q) = self.encode_contrastive_with_state(query, &mut state) else {
            return Vec::new();
        };
        let mut scored: Vec<(usize, f32)> = tool_descriptions
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let e = self.encode_contrastive_with_state(d, &mut state)?;
                // Both sides are already unit-norm, so the dot product is cosine.
                Some((i, q.iter().zip(e.iter()).map(|(a, b)| a * b).sum::<f32>()))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);
        scored
    }
}

/// Per-token byte table in the `(id, bytes)` form `ConstrainedDecoder` expects.
fn byte_table(tok: &SpTokenizer) -> Vec<(u32, Vec<u8>)> {
    tok.token_bytes()
        .into_iter()
        .enumerate()
        .map(|(i, b)| (i as u32, b))
        .collect()
}

/// The text a new token added, given the ids so far.
///
/// SentencePiece decoding is not per-token concatenative — byte-fallback pieces
/// only form a character together, and the dummy prefix is stripped from the
/// front — so a streaming caller has to diff decoded prefixes.
fn incremental_piece(tok: &SpTokenizer, ids: &[u32]) -> String {
    if ids.len() == 1 {
        return tok.decode(ids);
    }
    let before = tok.decode(&ids[..ids.len() - 1]);
    let after = tok.decode(ids);
    after
        .strip_prefix(&before)
        .map(str::to_string)
        .unwrap_or(after)
}

fn extract_between(text: &str, open: &str, close: &str) -> Option<String> {
    let start = text.find(open)? + open.len();
    let rest = &text[start..];
    Some(match rest.find(close) {
        Some(end) => rest[..end].to_string(),
        // Unterminated (hit the token cap) — return what there is.
        None => rest.to_string(),
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature sampling over the full distribution.
fn sample(logits: &[f32], temperature: f32, rng: &mut SplitMix64) -> u32 {
    let inv_t = 1.0 / temperature.max(1e-6);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f64;
    // One pass for the partition function, a second to walk it — avoids holding a
    // vocab-sized buffer of exponentials.
    for &v in logits {
        total += f64::from(((v - max) * inv_t).exp());
    }
    let mut target = rng.next_f64() * total;
    for (i, &v) in logits.iter().enumerate() {
        target -= f64::from(((v - max) * inv_t).exp());
        if target <= 0.0 {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// SplitMix64 — small, deterministic, and adequate for token sampling.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_json_removes_only_insignificant_whitespace() {
        // Whitespace between tokens goes; whitespace inside strings stays.
        assert_eq!(compact_json("{ \"a\" : 1 }"), "{\"a\":1}");
        assert_eq!(compact_json("[\n  1,\n  2\n]"), "[1,2]");
        assert_eq!(
            compact_json("{\"a\": \"keep  me\"}"),
            "{\"a\":\"keep  me\"}"
        );
        assert_eq!(
            compact_json("{\"a\": \"tab\\there\"}"),
            "{\"a\":\"tab\\there\"}"
        );
        // An escaped quote must not end the string early.
        assert_eq!(
            compact_json("{\"a\": \"q \\\" x\"}"),
            "{\"a\":\"q \\\" x\"}"
        );
        // A backslash before the closing quote is itself escaped.
        assert_eq!(compact_json("{\"a\": \"b\\\\\"}"), "{\"a\":\"b\\\\\"}");
        // Already compact: unchanged, so parity fixtures cannot move.
        let compact = "[{\"name\":\"f\",\"parameters\":{\"a\":1}}]";
        assert_eq!(compact_json(compact), compact);
        // Empty and degenerate input passes through.
        assert_eq!(compact_json(""), "");
        assert_eq!(compact_json("[]"), "[]");
    }

    /// The pretty and compact forms of one schema must produce the same prompt,
    /// because the model's answer depends on it.
    #[test]
    fn prompt_is_insensitive_to_schema_indentation() {
        let pretty = r#"[{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  }
}]"#;
        let compact = r#"[{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
        assert_eq!(
            V2Engine::build_prompt("q", pretty, None),
            V2Engine::build_prompt("q", compact, None)
        );
    }

    #[test]
    fn prompt_matches_upstream_template() {
        let p = V2Engine::build_prompt("hi", "[]", None);
        assert_eq!(
            p,
            "<|im_start|>user\n<tools>[]</tools>\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn prompt_includes_system_turn_first() {
        let p = V2Engine::build_prompt("q", "[]", Some("be terse"));
        assert!(p.starts_with("<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\n"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn extracts_tool_call_and_thinking() {
        let t = "<think>reasoning here</think>\n<tool_call>[{\"name\":\"f\"}]</tool_call>";
        assert_eq!(
            extract_between(t, THINK_START, THINK_END),
            Some("reasoning here".into())
        );
        assert_eq!(
            extract_between(t, TOOL_CALL_START, TOOL_CALL_END),
            Some("[{\"name\":\"f\"}]".into())
        );
        assert_eq!(
            extract_between("no markers", TOOL_CALL_START, TOOL_CALL_END),
            None
        );
    }

    /// A run that hit the token cap mid-call should still surface the partial
    /// payload rather than dropping it.
    #[test]
    fn extracts_unterminated_tool_call() {
        let t = "<tool_call>[{\"name\":\"get_wea";
        assert_eq!(
            extract_between(t, TOOL_CALL_START, TOOL_CALL_END),
            Some("[{\"name\":\"get_wea".into())
        );
    }

    #[test]
    fn argmax_picks_the_largest() {
        assert_eq!(argmax(&[0.1, 9.0, 2.0]), 1);
        assert_eq!(argmax(&[-5.0, -1.0, -9.0]), 1);
        // Ties resolve to the first, matching numpy's argmax.
        assert_eq!(argmax(&[1.0, 1.0]), 0);
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        let a: Vec<u32> = {
            let mut r = SplitMix64::new(7);
            (0..16).map(|_| sample(&logits, 0.8, &mut r)).collect()
        };
        let b: Vec<u32> = {
            let mut r = SplitMix64::new(7);
            (0..16).map(|_| sample(&logits, 0.8, &mut r)).collect()
        };
        assert_eq!(a, b);
        let c: Vec<u32> = {
            let mut r = SplitMix64::new(8);
            (0..16).map(|_| sample(&logits, 0.8, &mut r)).collect()
        };
        assert_ne!(a, c, "a different seed should give a different stream");
    }

    /// At a very low temperature sampling must collapse onto the argmax.
    #[test]
    fn low_temperature_sampling_approaches_argmax() {
        let mut logits = vec![0.0f32; 32];
        logits[19] = 10.0;
        let mut r = SplitMix64::new(1);
        for _ in 0..32 {
            assert_eq!(sample(&logits, 0.01, &mut r), 19);
        }
    }

    #[test]
    fn sampling_stays_in_range() {
        let logits = vec![0.0f32; 10];
        let mut r = SplitMix64::new(3);
        for _ in 0..200 {
            assert!(sample(&logits, 1.0, &mut r) < 10);
        }
    }

    #[test]
    fn splitmix_covers_the_unit_interval() {
        let mut r = SplitMix64::new(42);
        let mut lo = false;
        let mut hi = false;
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
            if v < 0.25 {
                lo = true;
            }
            if v > 0.75 {
                hi = true;
            }
        }
        assert!(lo && hi);
    }
}
