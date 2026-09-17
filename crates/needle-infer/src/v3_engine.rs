//! Needle 3 generation.
//!
//! Prompt assembly is shared with v2 ([`crate::prompt`]) because upstream
//! assembles it identically. What differs is the completion: v3 emits
//! `<think>` chain-of-thought before the call, which v2 did not, so the
//! default token budget is larger and the extractor has to look past it.

use needle_core::v3::{V3Cache, V3Model};

use crate::cact::CactV3;
use crate::prompt::{build_prompt, IM_END, THINK_END, THINK_START, TOOL_CALL_END, TOOL_CALL_START};
use crate::sp_tokenizer::SpTokenizer;
use crate::v3::{confidence_head, model_from_cact, V3LoadError};
use needle_core::v3::heads::ProbeHead;

/// Default generation cap.
///
/// Larger than v2's 128 because v3 reasons before answering: on the shipped
/// checkpoint a simple weather query spends ~20 tokens inside `<think>` before
/// opening `<tool_call>`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 256;

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The end-of-sequence token.
    Eos,
    /// The assistant turn closed with `<|im_end|>`.
    ImEnd,
    /// The token budget ran out.
    MaxTokens,
    /// The context window ran out.
    MaxSeqLen,
}

/// Generation settings.
#[derive(Debug, Clone)]
pub struct V3Options {
    pub max_new_tokens: usize,
    /// `0.0` is greedy.
    pub temperature: f32,
    pub seed: u64,
    pub system: Option<String>,
}

impl Default for V3Options {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            temperature: 0.0,
            seed: 0,
            system: None,
        }
    }
}

/// What a generation produced.
#[derive(Debug, Clone)]
pub struct V3Result {
    /// The decoded completion, without the prompt.
    pub text: String,
    pub tokens: Vec<u32>,
    pub stop: StopReason,
    /// Positions consumed, prompt included.
    pub positions: usize,
}

/// A loaded Needle 3 model with its tokenizer.
pub struct V3Engine {
    pub model: V3Model,
    pub tokenizer: SpTokenizer,
    /// Present when the container exports one. v3 exports confidence only —
    /// there is no contrastive head, so no retrieval.
    pub confidence: Option<ProbeHead>,
    eos_id: u32,
    bos_id: u32,
    im_end_id: Option<u32>,
}

impl V3Engine {
    /// Load from a `.cact` v3 container.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, V3EngineError> {
        let cact = CactV3::load(path).map_err(V3EngineError::Io)?;
        Self::from_cact(&cact)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, V3EngineError> {
        let cact = CactV3::from_bytes(bytes).map_err(|e| V3EngineError::Load(e.into()))?;
        Self::from_cact(&cact)
    }

    pub fn from_cact(cact: &CactV3) -> Result<Self, V3EngineError> {
        let model = model_from_cact(cact).map_err(V3EngineError::Load)?;
        let blob = cact.tokenizer_blob().ok_or(V3EngineError::NoTokenizer)?;
        let tokenizer = SpTokenizer::from_blob(blob).map_err(|_| V3EngineError::BadTokenizer)?;
        let im_end_id = tokenizer.id_of(IM_END);
        let confidence = confidence_head(cact, &model.cfg).map_err(V3EngineError::Load)?;
        Ok(Self {
            model,
            tokenizer,
            confidence,
            // Upstream fixes these in `needle/model/tokenizer.py`.
            eos_id: 1,
            bos_id: 2,
            im_end_id,
        })
    }

    /// Assemble the chat prompt. Shared with v2.
    pub fn build_prompt(query: &str, tools_json: &str, system: Option<&str>) -> String {
        build_prompt(query, tools_json, system)
    }

    /// Generate a completion, calling `on_token` with each decoded delta.
    ///
    /// The callback receives *decoded text*, not the raw SentencePiece piece.
    /// That distinction matters: a newline arrives as the byte-fallback token
    /// `<0x0A>`, and a caller appending raw pieces would render that literally.
    /// Concatenating every delta reproduces the returned text exactly.
    pub fn generate_with<F>(
        &self,
        query: &str,
        tools_json: &str,
        opts: &V3Options,
        mut on_token: F,
    ) -> V3Result
    where
        F: FnMut(u32, &str),
    {
        let prompt = build_prompt(query, tools_json, opts.system.as_deref());
        let mut ids = alloc_prompt_ids(self.bos_id, &self.tokenizer, &prompt);

        let budget = self
            .model
            .cfg
            .max_seq_len
            .saturating_sub(ids.len())
            .min(opts.max_new_tokens);
        let mut cache = V3Cache::new(&self.model.cfg, ids.len() + budget);

        // Prefill: every prompt token but the last only fills the cache. The
        // last one produces the first prediction.
        let mut logits = Vec::new();
        for &tok in &ids {
            logits = self.model.decode_step(&mut cache, tok);
        }

        let mut out_tokens = Vec::new();
        let mut emitted = String::new();
        let mut rng = SplitMix64::new(opts.seed);
        let mut stop = StopReason::MaxTokens;

        for _ in 0..budget {
            let next = if opts.temperature <= 0.0 {
                argmax(&logits)
            } else {
                sample(&logits, opts.temperature, &mut rng)
            };

            if next == self.eos_id {
                stop = StopReason::Eos;
                break;
            }
            if Some(next) == self.im_end_id {
                stop = StopReason::ImEnd;
                break;
            }

            out_tokens.push(next);
            // Decode the whole run and emit only what is new. Byte-fallback
            // tokens make up a character across several ids, so a per-piece
            // decode would split multi-byte text; this cannot.
            let full = self.tokenizer.decode(&out_tokens);
            if full.len() > emitted.len() {
                on_token(next, &full[emitted.len()..]);
                emitted = full;
            }
            ids.push(next);

            if cache.pos() >= self.model.cfg.max_seq_len {
                stop = StopReason::MaxSeqLen;
                break;
            }
            logits = self.model.decode_step(&mut cache, next);
        }

        V3Result {
            text: self.tokenizer.decode(&out_tokens),
            tokens: out_tokens,
            stop,
            positions: cache.pos(),
        }
    }

    /// Generate a completion.
    pub fn generate(&self, query: &str, tools_json: &str, opts: &V3Options) -> V3Result {
        self.generate_with(query, tools_json, opts, |_, _| {})
    }

    /// Generate with defaults and return the completion text.
    pub fn run(&self, query: &str, tools_json: &str) -> String {
        self.generate(query, tools_json, &V3Options::default()).text
    }

    /// The tool-call payload, or `None` when the model emitted no call.
    ///
    /// `Some("[]")` is a deliberate abstention — the model decided no tool
    /// applies — and is different from `None`, which means no `<tool_call>`
    /// markers were produced at all. Callers that conflate the two will treat
    /// a considered "no" as a failure.
    pub fn run_json(&self, query: &str, tools_json: &str) -> Option<String> {
        extract_tool_call(&self.run(query, tools_json))
    }

    /// How confident the model is in a completion it produced.
    ///
    /// **Pass the completion.** The head scores a finished judgement — the
    /// formatted prompt *plus* the answer — not a question.
    ///
    /// v3 differs from v2 here, and the difference is a trap. On v2 a bare
    /// query scored near zero, so the misuse announced itself; on the shipped
    /// v3 checkpoint the same query scores 0.80 while the correct completion
    /// scores 0.93 and a wrong one 0.26. A bare query therefore looks like a
    /// confident answer and is not one. Measured, not assumed — see
    /// `confidence_scores_the_answer_not_the_question`.
    ///
    /// Returns a probability in `(0, 1)`, or `None` when the container exports
    /// no confidence head.
    pub fn confidence_for(&self, query: &str, tools_json: &str, completion: &str) -> Option<f32> {
        let head = self.confidence.as_ref()?;
        let mut text = build_prompt(query, tools_json, None);
        text.push_str(completion);

        let mut ids = Vec::with_capacity(text.len() / 3 + 2);
        ids.push(self.bos_id);
        ids.extend(self.tokenizer.encode(&text));
        ids.truncate(self.model.cfg.max_seq_len);

        let cells = self.model.forward_cells(&ids);
        let logit = head.forward(&cells, ids.len(), self.model.cfg.d_model)[0];
        Some(1.0 / (1.0 + (-logit).exp()))
    }

    /// Generate, then score what was generated.
    ///
    /// Cheaper to reason about than calling [`Self::generate`] and
    /// [`Self::confidence_for`] separately, and impossible to get the argument
    /// order wrong.
    pub fn run_scored(&self, query: &str, tools_json: &str) -> (V3Result, Option<f32>) {
        let res = self.generate(query, tools_json, &V3Options::default());
        let p = self.confidence_for(query, tools_json, &res.text);
        (res, p)
    }

    /// The model's reasoning, when it emitted any.
    pub fn reasoning(text: &str) -> Option<&str> {
        between(text, THINK_START, THINK_END).map(str::trim)
    }
}

/// Pull the `<tool_call>…</tool_call>` payload out of a completion.
pub fn extract_tool_call(text: &str) -> Option<String> {
    between(text, TOOL_CALL_START, TOOL_CALL_END).map(|s| s.trim().to_string())
}

fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.find(open)? + open.len();
    let rest = &text[start..];
    match rest.find(close) {
        Some(end) => Some(&rest[..end]),
        // An unterminated marker still carries the payload the model got to
        // before the budget ran out; returning it beats returning nothing.
        None => Some(rest),
    }
}

fn alloc_prompt_ids(bos: u32, tok: &SpTokenizer, prompt: &str) -> Vec<u32> {
    let mut ids = Vec::with_capacity(prompt.len() / 3 + 2);
    ids.push(bos);
    ids.extend(tok.encode(prompt));
    ids
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut top = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > top {
            top = v;
            best = i;
        }
    }
    best as u32
}

fn sample(logits: &[f32], temperature: f32, rng: &mut SplitMix64) -> u32 {
    let inv = 1.0 / temperature;
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) * inv).exp() as f64;
    }
    let mut target = rng.next_f64() * sum;
    for (i, &v) in logits.iter().enumerate() {
        target -= ((v - max) * inv).exp() as f64;
        if target <= 0.0 {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// SplitMix64 — small, seedable and reproducible, which is what sampling
/// parity needs. Not cryptographic.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Anything that can go wrong loading an engine.
#[derive(Debug)]
pub enum V3EngineError {
    Io(std::io::Error),
    Load(V3LoadError),
    NoTokenizer,
    BadTokenizer,
}

impl core::fmt::Display for V3EngineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Load(e) => write!(f, "{e}"),
            Self::NoTokenizer => write!(f, "container carries no tokenizer"),
            Self::BadTokenizer => write!(f, "embedded tokenizer did not decode"),
        }
    }
}

impl std::error::Error for V3EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_a_call_past_the_reasoning() {
        let text = "<think>\nweather in Paris\n</think>\n\
                    <tool_call>[{\"name\":\"get_weather\"}]</tool_call>";
        assert_eq!(
            extract_tool_call(text).as_deref(),
            Some("[{\"name\":\"get_weather\"}]")
        );
        assert_eq!(V3Engine::reasoning(text), Some("weather in Paris"));
    }

    #[test]
    fn an_empty_array_is_an_abstention_not_a_failure() {
        // The distinction callers must not collapse.
        assert_eq!(
            extract_tool_call("<tool_call>[]</tool_call>").as_deref(),
            Some("[]")
        );
        assert_eq!(extract_tool_call("I cannot help with that"), None);
    }

    #[test]
    fn an_unterminated_call_still_yields_its_payload() {
        // Budget exhausted mid-call: better to hand back what there is than
        // to report nothing.
        assert_eq!(
            extract_tool_call("<tool_call>[{\"name\":\"get_w").as_deref(),
            Some("[{\"name\":\"get_w")
        );
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let logits = [0.1f32, 2.0, 0.3, 1.5];
        let draw = |seed| {
            let mut r = SplitMix64::new(seed);
            (0..16)
                .map(|_| sample(&logits, 1.0, &mut r))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(42), draw(42), "same seed must replay");
        assert_ne!(draw(42), draw(43), "different seeds should diverge");
    }

    #[test]
    fn greedy_picks_the_maximum() {
        assert_eq!(argmax(&[0.1, 2.0, 0.3]), 1);
        assert_eq!(argmax(&[-5.0, -1.0, -9.0]), 1);
    }
}
