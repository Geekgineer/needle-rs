//! High-level Needle inference engine.
//!
//! Orchestrates: SafeTensors load → model construction → encode → greedy decode.

use crate::constrained::{ConstrainedDecoder, ToolDef};
use crate::safetensors::SafeTensors;
use crate::tokenizer::{Vocabulary, EOS_ID, TOOLS_ID};
use needle_core::{
    config::FfnActivation,
    ffn::FfnWeights,
    layers::{DecoderLayer, EncoderLayer},
    model::NeedleModel,
    quant::QuantizedWeight,
    TransformerConfig,
};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub token_ids: Vec<u32>,
    pub text: String,
}

/// Contrastive projection head for query/tool embedding (mirrors Python `encode_contrastive`).
///
/// Two-layer MLP: relu(x @ hidden_kernel + hidden_bias) @ proj_kernel, then L2-normalized.
/// Matches Python `SimpleAttentionNetwork.encode_contrastive()` exactly.
struct ContrastiveHead {
    hidden_kernel: Vec<f32>, // [d_model, hidden_dim] row-major (in × out)
    hidden_bias: Vec<f32>,   // [hidden_dim]
    proj_kernel: Vec<f32>,   // [hidden_dim, contrastive_dim] row-major
    hidden_dim: usize,
    pub contrastive_dim: usize,
}

impl ContrastiveHead {
    /// Project a mean-pooled encoder vector to a L2-normalized contrastive embedding.
    #[allow(clippy::needless_range_loop)]
    fn encode(&self, pooled: &[f32]) -> Vec<f32> {
        let d = pooled.len();
        let h = self.hidden_dim;
        let c = self.contrastive_dim;

        // hidden = relu(pooled @ hidden_kernel + hidden_bias)
        let mut hidden = vec![0.0f32; h];
        for j in 0..h {
            let mut acc = self.hidden_bias[j];
            for i in 0..d {
                acc += pooled[i] * self.hidden_kernel[i * h + j];
            }
            hidden[j] = acc.max(0.0); // ReLU
        }

        // proj = hidden @ proj_kernel (no bias)
        let mut proj = vec![0.0f32; c];
        for j in 0..c {
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += hidden[i] * self.proj_kernel[i * c + j];
            }
            proj[j] = acc;
        }

        // L2-normalize: proj / sqrt(sum(proj²) + ε)  — matches Python safe_norm with eps=1e-12
        let sq_sum: f32 = proj.iter().map(|x| x * x).sum();
        let norm = (sq_sum + 1e-12_f32).sqrt();
        proj.iter_mut().for_each(|x| *x /= norm);
        proj
    }
}

pub struct NeedleEngine {
    model: NeedleModel,
    vocab: Vocabulary,
    max_gen_len: usize,
    /// Optional contrastive head for query/tool retrieval embedding.
    /// Present only when the SafeTensors file contains `contrastive_proj_kernel`.
    contrastive: Option<ContrastiveHead>,
}

impl NeedleEngine {
    /// Expose the underlying model for direct encode/decode_step access in tests.
    pub fn model(&self) -> &NeedleModel {
        &self.model
    }

    /// Expose config for test helpers.
    pub fn cfg(&self) -> &needle_core::TransformerConfig {
        &self.model.cfg
    }

    /// Load model from a SafeTensors weight file and a vocabulary text file.
    pub fn load<P: AsRef<Path>>(weights_path: P, vocab_path: P) -> std::io::Result<Self> {
        let st = SafeTensors::load(weights_path)?;
        let vocab = Vocabulary::load_text(vocab_path)?;
        Self::from_parts(st, vocab)
    }

    /// Load model from in-memory buffers (WASM, embedding, network fetch).
    /// `weights_bytes`: raw SafeTensors file bytes.
    /// `vocab_text`:    vocabulary text (one piece per line).
    pub fn from_bytes(weights_bytes: Vec<u8>, vocab_text: &str) -> std::io::Result<Self> {
        let st = SafeTensors::from_bytes(weights_bytes)?;
        let vocab = Vocabulary::parse(vocab_text);
        Self::from_parts(st, vocab)
    }

    fn from_parts(st: SafeTensors, vocab: Vocabulary) -> std::io::Result<Self> {
        let cfg = load_config_from_safetensors(&st);
        cfg.validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let model = load_model(&st, &cfg)?;
        let contrastive = load_contrastive_head(&st, cfg.d_model);
        Ok(Self {
            model,
            vocab,
            max_gen_len: cfg.max_dec_len,
            contrastive,
        })
    }

    /// Run single-example inference. Equivalent to Python `generate(stream=False)`.
    pub fn run(&self, query: &str, tools_json: &str) -> InferenceResult {
        self.run_impl(query, tools_json, |_, _| {})
    }

    /// Run inference with a per-token streaming callback.
    ///
    /// `on_token(token_id, piece_text)` fires for each generated token in order,
    /// including the initial `<tool_call>` sentinel. Matches Python `generate(stream=True)`.
    /// The returned `InferenceResult.text` is the same post-processed string as `run()`.
    pub fn run_stream<F>(&self, query: &str, tools_json: &str, on_token: F) -> InferenceResult
    where
        F: FnMut(u32, &str),
    {
        self.run_impl(query, tools_json, on_token)
    }

    /// Run inference on multiple examples sequentially.
    ///
    /// Equivalent to calling `run()` for each example in order. Matches Python `generate_batch`.
    pub fn run_batch(&self, examples: &[(&str, &str)]) -> Vec<InferenceResult> {
        examples.iter().map(|(q, t)| self.run(q, t)).collect()
    }

    /// Encode `text` to a L2-normalized contrastive embedding (`contrastive_dim` floats).
    ///
    /// Returns `None` if the model was loaded without a contrastive head
    /// (i.e., the SafeTensors file does not contain `contrastive_proj_kernel`).
    ///
    /// Mirrors Python `encode_for_retrieval` / `encode_contrastive`:
    ///   tokenize → encode → mean-pool → relu(hidden) → proj → L2-normalize.
    pub fn encode_contrastive(&self, text: &str) -> Option<Vec<f32>> {
        let head = self.contrastive.as_ref()?;
        let d = self.model.cfg.d_model;

        let token_ids = self.vocab.encode(text);
        if token_ids.is_empty() {
            return Some(vec![0.0f32; head.contrastive_dim]);
        }

        // Encode (no tools separator — pure text embedding)
        let mut enc_kv = self.model.make_enc_kv_caches(token_ids.len());
        let enc_hidden = self.model.encode(&token_ids, &mut enc_kv);

        // Mean-pool over all positions: since encoder input is never padded in Rust
        // inference, this is a simple mean across the sequence.
        let seq_len = token_ids.len();
        let mut pooled = vec![0.0f32; d];
        for t in 0..seq_len {
            for j in 0..d {
                pooled[j] += enc_hidden[t * d + j];
            }
        }
        let inv_n = 1.0 / seq_len as f32;
        pooled.iter_mut().for_each(|x| *x *= inv_n);

        Some(head.encode(&pooled))
    }

    /// Dimension of contrastive embeddings (0 if no contrastive head loaded).
    pub fn contrastive_dim(&self) -> usize {
        self.contrastive.as_ref().map_or(0, |h| h.contrastive_dim)
    }

    /// Rank `tool_descriptions` by contrastive similarity to `query`.
    ///
    /// Returns up to `top_k` `(index, score)` pairs sorted by descending cosine similarity.
    /// Both embeddings are L2-normalized so dot product equals cosine similarity.
    /// Returns an empty Vec if the model has no contrastive head.
    ///
    /// Mirrors Python `retrieve_tools(query, tools, top_k)` from `run.py`.
    pub fn retrieve_tools(
        &self,
        query: &str,
        tool_descriptions: &[&str],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let q_emb = match self.encode_contrastive(query) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut scores: Vec<(usize, f32)> = tool_descriptions
            .iter()
            .enumerate()
            .filter_map(|(i, desc)| {
                let t_emb = self.encode_contrastive(desc)?;
                let score: f32 = q_emb.iter().zip(t_emb.iter()).map(|(a, b)| a * b).sum();
                Some((i, score))
            })
            .collect();

        scores.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        scores.truncate(top_k);
        scores
    }

    // ── Core encode + decode ─────────────────────────────────────────────────

    /// Shared implementation for `run` and `run_stream`.
    ///
    /// `on_token(token_id, piece_text)` fires for every generated token (including
    /// `<tool_call>`) before post-processing. The returned `InferenceResult.text`
    /// has `<tool_call>` stripped and original tool names restored.
    fn run_impl<F>(&self, query: &str, tools_json: &str, mut on_token: F) -> InferenceResult
    where
        F: FnMut(u32, &str),
    {
        // Normalize tool names to snake_case before encoding — matches Python run.py normalize_tools.
        let tool_defs = ToolDef::from_json(tools_json);
        let normalized_tools = normalize_tools_json(tools_json, &tool_defs);

        // Compact tools JSON — Python re-serializes via json.dumps(separators=(",",":")),
        // so the encoder always sees whitespace-free JSON regardless of caller formatting.
        let compact_tools = compact_json(&normalized_tools);

        // Tokenize
        let query_ids = self.vocab.encode(query);
        let tools_ids = self.vocab.encode(&compact_tools);

        // Guard: if both query and tools are empty, the encoder would see only
        // [TOOLS_ID] — a single token with no meaning. Return early.
        if query_ids.is_empty() && tools_ids.is_empty() {
            return InferenceResult {
                token_ids: Vec::new(),
                text: String::new(),
            };
        }

        // Match Python's _build_encoder_input truncation:
        // query capped at max_enc-2 to guarantee TOOLS_ID + ≥1 tool token always fit.
        let max_enc = self.model.cfg.max_enc_len;
        if max_enc == 0 {
            return InferenceResult {
                token_ids: Vec::new(),
                text: String::new(),
            };
        }
        let q_len = query_ids.len().min(max_enc.saturating_sub(2));
        let remaining = max_enc.saturating_sub(q_len + 1);
        let t_len = tools_ids.len().min(remaining);

        let mut enc_input = Vec::with_capacity(q_len + 1 + t_len);
        enc_input.extend_from_slice(&query_ids[..q_len]);
        enc_input.push(TOOLS_ID);
        enc_input.extend_from_slice(&tools_ids[..t_len]);

        // Allocate KV caches and encode
        let enc_len = enc_input.len();
        let mut enc_kv = self.model.make_enc_kv_caches(enc_len);
        let mut dec_kv = self.model.make_dec_kv_caches();
        self.model.encode(&enc_input, &mut enc_kv);

        // Build token byte map for constrained decoder trie
        let token_bytes: Vec<(u32, Vec<u8>)> = self
            .vocab
            .id_to_piece
            .iter()
            .enumerate()
            .map(|(i, piece)| (i as u32, piece.replace('▁', " ").into_bytes()))
            .collect();
        let mut constrained = ConstrainedDecoder::new(&tool_defs, token_bytes);

        // Greedy decode starting from [EOS]
        let mut output_ids = Vec::with_capacity(64);
        let mut current_token = EOS_ID;
        let mut logits = vec![0.0f32; self.model.cfg.vocab_size];

        for _step in 0..self.max_gen_len {
            self.model
                .decode_step(current_token, &enc_kv, &mut dec_kv, &mut logits);

            let mask = constrained.logit_mask(self.model.cfg.vocab_size);
            for (l, &m) in logits.iter_mut().zip(mask.iter()) {
                *l += m;
            }

            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap_or(EOS_ID);

            if next_token == EOS_ID {
                break;
            }

            // Fire streaming callback with the raw piece text for this token.
            let piece = self
                .vocab
                .id_to_piece
                .get(next_token as usize)
                .map(|p| p.replace('▁', " "))
                .unwrap_or_default();
            on_token(next_token, &piece);

            output_ids.push(next_token);
            current_token = next_token;
            constrained.update(next_token);
        }

        let raw = self.vocab.decode_ids(&output_ids);

        // Strip <tool_call> sentinel — Python run.py strips this prefix before returning.
        let tool_call_piece = self
            .vocab
            .id_to_piece
            .get(crate::tokenizer::TOOL_CALL_ID as usize)
            .map(|p| p.replace('▁', " "))
            .unwrap_or_default();
        let stripped = if !tool_call_piece.is_empty() {
            raw.strip_prefix(&tool_call_piece).unwrap_or(&raw)
        } else {
            &raw
        };

        // Restore original tool names.
        let text = restore_tool_names(stripped, &tool_defs);

        InferenceResult {
            token_ids: output_ids,
            text,
        }
    }
}

/// Replace each tool's original name with its snake_case form in the JSON string.
/// Matches Python run.py `normalize_tools`: the encoder must see snake_case names
/// since the model was trained on normalized tool names.
///
/// Handles both compact JSON (`"name":"Foo"`) and spaced JSON (`"name": "Foo"`).
fn normalize_tools_json(json: &str, tool_defs: &[crate::constrained::ToolDef]) -> String {
    // Build a lookup: original name → snake name (only for names that differ).
    let renames: Vec<(&str, &str)> = tool_defs
        .iter()
        .filter(|t| t.name != t.snake_name)
        .map(|t| (t.name.as_str(), t.snake_name.as_str()))
        .collect();

    if renames.is_empty() {
        return json.to_string();
    }

    // Walk the JSON bytes looking for `"name"` keys and replace their string values.
    // All pattern characters ("name", ':', '"', whitespace) are ASCII, so byte-level
    // scanning is valid. Non-matching regions are copied as string slices (not byte
    // by byte) so multi-byte UTF-8 sequences in descriptions/values are preserved.
    let bytes = json.as_bytes();
    let mut out = String::with_capacity(json.len() + 16);
    let mut i = 0;
    let mut last_flush = 0; // byte index up to which `json` has been pushed to `out`

    while i < bytes.len() {
        // Detect `"name"` — all ASCII, safe to match on bytes.
        if bytes[i] == b'"' && bytes[i..].starts_with(b"\"name\"") {
            // Flush everything before this "name" key as a raw UTF-8 slice.
            out.push_str(&json[last_flush..i]);

            out.push_str("\"name\"");
            i += 6;

            // Skip optional whitespace, then ':'
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b':' {
                out.push(':');
                i += 1;
            }

            // Skip optional whitespace, then opening '"'
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'"' {
                out.push('"');
                i += 1;
                let val_start = i;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    } // skip escape sequence
                    if bytes[i] == b'"' {
                        break;
                    }
                    i += 1;
                }
                let val = &json[val_start..i];
                let replacement = renames
                    .iter()
                    .find(|(orig, _)| *orig == val)
                    .map(|(_, snake)| *snake)
                    .unwrap_or(val);
                out.push_str(replacement);
                if i < bytes.len() {
                    out.push('"');
                    i += 1;
                }
            }

            last_flush = i;
        } else {
            i += 1;
        }
    }

    // Flush any remaining bytes (includes non-ASCII UTF-8 in descriptions etc.)
    out.push_str(&json[last_flush..]);
    out
}

/// Strip all whitespace outside JSON string literals, producing compact JSON.
///
/// Mirrors Python `json.dumps(tools, separators=(",", ":"))`:
/// Python's `normalize_tools` always re-serializes through json.dumps, so the encoder
/// always receives compact JSON regardless of the caller's original formatting.
/// Rust must do the same so encoder inputs match for any whitespace variant.
fn compact_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    let mut in_string = false;

    while let Some(ch) = chars.next() {
        if in_string {
            out.push(ch);
            if ch == '\\' {
                // Copy the escaped character verbatim (handles \", \\, \uXXXX, etc.)
                if let Some(escaped) = chars.next() {
                    out.push(escaped);
                }
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
            out.push('"');
        } else if ch.is_ascii_whitespace() {
            // Drop structural whitespace outside strings
        } else {
            out.push(ch);
        }
    }
    out
}

/// Replace snake_case tool names in the model's output JSON with the caller's original names.
///
/// Mirrors Python run.py `restore_tool_names`: scans for `"name":"snake"` in the output
/// and rewrites to `"name":"original"`. Longest snake names replaced first to avoid
/// partial matches (same order as Python's fallback path).
fn restore_tool_names(text: &str, tool_defs: &[crate::constrained::ToolDef]) -> String {
    // Build lookup: snake_name → original name (only where they differ).
    // Sort longest-snake-name first to prevent partial matches on shared prefixes.
    let mut renames: Vec<(&str, &str)> = tool_defs
        .iter()
        .filter(|t| t.name != t.snake_name)
        .map(|t| (t.snake_name.as_str(), t.name.as_str()))
        .collect();
    renames.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()));

    if renames.is_empty() {
        return text.to_string();
    }

    // Byte-level scan identical to normalize_tools_json but in reverse direction.
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut last_flush = 0;

    while i < bytes.len() {
        if bytes[i] == b'"' && bytes[i..].starts_with(b"\"name\"") {
            out.push_str(&text[last_flush..i]);
            out.push_str("\"name\"");
            i += 6;

            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b':' {
                out.push(':');
                i += 1;
            }
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }

            if i < bytes.len() && bytes[i] == b'"' {
                out.push('"');
                i += 1;
                let val_start = i;
                while i < bytes.len() {
                    if bytes[i] == b'\\' {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        break;
                    }
                    i += 1;
                }
                let val = &text[val_start..i];
                let replacement = renames
                    .iter()
                    .find(|(snake, _)| *snake == val)
                    .map(|(_, orig)| *orig)
                    .unwrap_or(val);
                out.push_str(replacement);
                if i < bytes.len() {
                    out.push('"');
                    i += 1;
                }
            }

            last_flush = i;
        } else {
            i += 1;
        }
    }

    out.push_str(&text[last_flush..]);
    out
}

/// Load the optional contrastive projection head from SafeTensors.
///
/// Present only if the export script wrote `contrastive_proj_kernel`.
/// Tensor shapes encode dimensions: hidden_kernel [d_model × hidden_dim], proj_kernel [hidden_dim × contrastive_dim].
fn load_contrastive_head(st: &SafeTensors, d_model: usize) -> Option<ContrastiveHead> {
    let proj_kernel = st.get_f32("contrastive_proj_kernel")?;
    let hidden_kernel = st.get_f32("contrastive_hidden_kernel")?;
    let hidden_bias = st.get_f32("contrastive_hidden_bias")?;

    let hidden_dim = hidden_bias.len();
    if hidden_dim == 0
        || hidden_kernel.len() != d_model * hidden_dim
        || proj_kernel.len() % hidden_dim != 0
    {
        eprintln!("[needle] contrastive head shape mismatch — skipping");
        return None;
    }
    let contrastive_dim = proj_kernel.len() / hidden_dim;

    Some(ContrastiveHead {
        hidden_kernel,
        hidden_bias,
        proj_kernel,
        hidden_dim,
        contrastive_dim,
    })
}

/// Extract model config from SafeTensors `__metadata__`, falling back to defaults.
/// The export script writes all config fields as string-valued metadata entries.
fn load_config_from_safetensors(st: &SafeTensors) -> TransformerConfig {
    let d = TransformerConfig::default();

    let usize_field = |key: &str, fallback: usize| -> usize {
        st.get_metadata(key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(fallback)
    };
    let f32_field = |key: &str, fallback: f32| -> f32 {
        st.get_metadata(key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(fallback)
    };
    let bool_field = |key: &str, fallback: bool| -> bool {
        st.get_metadata(key)
            .map(|s| matches!(s, "True" | "true" | "1"))
            .unwrap_or(fallback)
    };

    // max_enc_len: new exports use "max_enc_len"; old exports used "max_seq_len".
    let max_enc_len = st
        .get_metadata("max_enc_len")
        .and_then(|s| s.parse().ok())
        .or_else(|| st.get_metadata("max_seq_len").and_then(|s| s.parse().ok()))
        .unwrap_or(d.max_enc_len);

    let activation = st
        .get_metadata("activation")
        .map(FfnActivation::parse)
        .unwrap_or(d.activation.clone());

    TransformerConfig {
        d_model: usize_field("d_model", d.d_model),
        num_heads: usize_field("num_heads", d.num_heads),
        num_kv_heads: usize_field("num_kv_heads", d.num_kv_heads),
        num_layers: usize_field("num_encoder_layers", d.num_layers),
        num_dec_layers: usize_field("num_decoder_layers", d.num_dec_layers),
        vocab_size: usize_field("vocab_size", d.vocab_size),
        max_enc_len,
        max_dec_len: usize_field("max_dec_len", d.max_dec_len),
        ffn_dim: usize_field("ffn_dim", d.ffn_dim),
        no_feedforward: bool_field("no_feedforward", d.no_feedforward),
        activation,
        rope_theta: f32_field("rope_theta", d.rope_theta),
        ..d
    }
}

/// Build NeedleModel from loaded SafeTensors tensors.
/// Tensor naming convention (set by export.py):
///   embedding:                   "embedding"
///   encoder layer i self-attn:   "encoder.{i}.self_attn.wq", ...
///   decoder layer i self-attn:   "decoder.{i}.self_attn.wq", ...
///   decoder layer i cross-attn:  "decoder.{i}.cross_attn.wq", ...
///   norms:                       "encoder.{i}.norm", "decoder.{i}.self_attn_norm", etc.
///   gates:                       "encoder.{i}.self_attn_gate", etc.
fn load_model(st: &SafeTensors, cfg: &TransformerConfig) -> std::io::Result<NeedleModel> {
    let d = cfg.d_model;
    let v = cfg.vocab_size;

    let embedding = st.get_f32("embedding").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing embedding tensor")
    })?;
    if embedding.len() != v * d {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "embedding size mismatch: got {}, expected {}",
                embedding.len(),
                v * d
            ),
        ));
    }

    let encoder_layers = (0..cfg.num_layers)
        .map(|i| load_encoder_layer(st, cfg, i))
        .collect();

    let decoder_layers = (0..cfg.num_dec_layers)
        .map(|i| load_decoder_layer(st, cfg, i))
        .collect();

    let encoder_final_norm = st
        .get_f32("encoder_final_norm")
        .unwrap_or_else(|| vec![0.0f32; d]);
    let decoder_final_norm = st
        .get_f32("decoder_final_norm")
        .unwrap_or_else(|| vec![0.0f32; d]);

    Ok(NeedleModel::new(
        cfg.clone(),
        embedding,
        encoder_layers,
        decoder_layers,
        encoder_final_norm,
        decoder_final_norm,
    ))
}

fn load_encoder_layer(st: &SafeTensors, cfg: &TransformerConfig, i: usize) -> EncoderLayer {
    let prefix = format!("encoder.{i}");
    let (ffn, ffn_gate, ffn_norm, ffn_activation) = if cfg.no_feedforward {
        (None, 0.0, None, None)
    } else {
        let f = load_ffn_weights(st, cfg, &format!("{prefix}.ffn"));
        let g = load_scalar(st, &format!("{prefix}.ffn_gate"));
        let n = load_vec(st, &format!("{prefix}.ffn_norm"), cfg.d_model);
        (Some(f), g, Some(n), Some(cfg.activation.clone()))
    };
    EncoderLayer {
        self_attn: load_attn_weights(st, cfg, &format!("{prefix}.self_attn")),
        self_attn_gate: load_scalar(st, &format!("{prefix}.self_attn_gate")),
        norm: load_vec(st, &format!("{prefix}.norm"), cfg.d_model),
        ffn,
        ffn_gate,
        ffn_norm,
        ffn_activation,
    }
}

fn load_decoder_layer(st: &SafeTensors, cfg: &TransformerConfig, i: usize) -> DecoderLayer {
    let prefix = format!("decoder.{i}");
    let (ffn, ffn_gate, ffn_norm, ffn_activation) = if cfg.no_feedforward {
        (None, 0.0, None, None)
    } else {
        let f = load_ffn_weights(st, cfg, &format!("{prefix}.ffn"));
        let g = load_scalar(st, &format!("{prefix}.ffn_gate"));
        let n = load_vec(st, &format!("{prefix}.ffn_norm"), cfg.d_model);
        (Some(f), g, Some(n), Some(cfg.activation.clone()))
    };
    DecoderLayer {
        self_attn: load_attn_weights(st, cfg, &format!("{prefix}.self_attn")),
        self_attn_gate: load_scalar(st, &format!("{prefix}.self_attn_gate")),
        self_attn_norm: load_vec(st, &format!("{prefix}.self_attn_norm"), cfg.d_model),
        cross_attn: load_attn_weights(st, cfg, &format!("{prefix}.cross_attn")),
        cross_attn_gate: load_scalar(st, &format!("{prefix}.cross_attn_gate")),
        cross_attn_norm: load_vec(st, &format!("{prefix}.cross_attn_norm"), cfg.d_model),
        ffn,
        ffn_gate,
        ffn_norm,
        ffn_activation,
    }
}

fn load_ffn_weights(st: &SafeTensors, cfg: &TransformerConfig, prefix: &str) -> FfnWeights {
    let d = cfg.d_model;
    let ff = cfg.ffn_dim;
    FfnWeights {
        gate_proj: load_quant(st, &format!("{prefix}.gate_proj"), d, ff),
        up_proj: load_quant(st, &format!("{prefix}.up_proj"), d, ff),
        down_proj: load_quant(st, &format!("{prefix}.down_proj"), ff, d),
    }
}

fn load_attn_weights(
    st: &SafeTensors,
    cfg: &TransformerConfig,
    prefix: &str,
) -> needle_core::attn::AttnWeights {
    let d = cfg.d_model;
    let h = cfg.num_heads;
    let kv_h = cfg.num_kv_heads;
    let hd = cfg.head_dim();

    needle_core::attn::AttnWeights {
        wq: load_quant(st, &format!("{prefix}.wq"), d, h * hd),
        wk: load_quant(st, &format!("{prefix}.wk"), d, kv_h * hd),
        wv: load_quant(st, &format!("{prefix}.wv"), d, kv_h * hd),
        wo: load_quant(st, &format!("{prefix}.wo"), h * hd, d),
        q_norm: load_vec(st, &format!("{prefix}.q_norm"), hd),
        k_norm: load_vec(st, &format!("{prefix}.k_norm"), hd),
    }
}

fn load_quant(st: &SafeTensors, name: &str, in_feat: usize, out_feat: usize) -> QuantizedWeight {
    // Try loading pre-quantized (data + scales), fall back to f32 + quantize on load
    let scale_name = format!("{name}.scale");
    if let (Some(raw), Some(scales)) = (st.get_raw(name), st.get_f32(&scale_name)) {
        if scales.len() % out_feat != 0 {
            eprintln!(
                "Warning: {name}.scale length {} is not divisible by out_feat {}; falling back to re-quantize",
                scales.len(), out_feat
            );
            let w = st
                .get_f32(name)
                .unwrap_or_else(|| vec![0.0f32; in_feat * out_feat]);
            return QuantizedWeight::quantize(&w, in_feat, out_feat);
        }
        let num_groups = scales.len() / out_feat;
        QuantizedWeight {
            data: raw.to_vec(),
            scales,
            in_feat,
            out_feat,
            num_groups,
        }
    } else {
        // Load as f32 and quantize
        let w = st.get_f32(name).unwrap_or_else(|| {
            eprintln!("Warning: missing tensor {name}, using zeros");
            vec![0.0f32; in_feat * out_feat]
        });
        QuantizedWeight::quantize(&w, in_feat, out_feat)
    }
}

fn load_vec(st: &SafeTensors, name: &str, len: usize) -> Vec<f32> {
    st.get_f32(name).unwrap_or_else(|| vec![0.0f32; len])
}

fn load_scalar(st: &SafeTensors, name: &str) -> f32 {
    st.get_f32(name)
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained::ToolDef;

    /// compact_json: whitespace outside strings is removed; string content is preserved.
    #[test]
    fn test_compact_json_strips_whitespace() {
        assert_eq!(compact_json(r#"{"a": 1, "b": 2}"#), r#"{"a":1,"b":2}"#);
        assert_eq!(
            compact_json(r#"[ { "name" : "Foo" } ]"#),
            r#"[{"name":"Foo"}]"#
        );
        // Whitespace INSIDE strings must be preserved
        assert_eq!(
            compact_json(r#"{"desc": "hello world"}"#),
            r#"{"desc":"hello world"}"#
        );
        // Escaped quote inside string must not end the string early
        assert_eq!(
            compact_json(r#"{"d": "say \"hi\""}"#),
            r#"{"d":"say \"hi\""}"#
        );
        // Already compact — no change
        assert_eq!(compact_json(r#"[{"name":"foo"}]"#), r#"[{"name":"foo"}]"#);
        // Multi-byte UTF-8 inside strings must not be corrupted (was broken with b as char)
        let with_utf8 = "{\"desc\": \"caf\u{00e9}\"}"; // {"desc": "café"}
        let compacted = compact_json(with_utf8);
        assert_eq!(compacted, "{\"desc\":\"caf\u{00e9}\"}"); // {"desc":"café"} — accent preserved
        assert!(
            compacted.contains('\u{00e9}'),
            "é must survive compaction, not be split into bytes"
        );
        // Backslash-escape sequence — the char after \ must not be re-processed
        assert_eq!(compact_json(r#"{"k": "a\\b"}"#), r#"{"k":"a\\b"}"#);
    }

    /// restore_tool_names: snake_case names in output are restored to original names.
    #[test]
    fn test_restore_tool_names_basic() {
        let tool_defs =
            ToolDef::from_json(r#"[{"name":"getWeather","description":"x","parameters":{}}]"#);
        // Model emits snake_case; we must get original back.
        let output = r#"[{"name":"get_weather","arguments":{"location":"Paris"}}]"#;
        let restored = restore_tool_names(output, &tool_defs);
        assert!(
            restored.contains("\"name\":\"getWeather\""),
            "original name not restored: {restored}"
        );
        assert!(
            !restored.contains("get_weather"),
            "snake_case leaked into output: {restored}"
        );
    }

    /// restore_tool_names: names already in snake_case (no mapping needed) pass through.
    #[test]
    fn test_restore_tool_names_already_snake() {
        let tool_defs =
            ToolDef::from_json(r#"[{"name":"get_weather","description":"x","parameters":{}}]"#);
        let output = r#"[{"name":"get_weather","arguments":{"location":"Paris"}}]"#;
        let restored = restore_tool_names(output, &tool_defs);
        assert_eq!(restored, output, "no-op rename must not modify output");
    }

    /// restore_tool_names: multiple tools, longest snake name replaced first.
    #[test]
    fn test_restore_tool_names_multiple_tools() {
        let tool_defs = ToolDef::from_json(
            r#"[
            {"name":"getWeather","description":"x","parameters":{}},
            {"name":"getWeatherForecast","description":"y","parameters":{}}
        ]"#,
        );
        let output = r#"[{"name":"get_weather_forecast","arguments":{}}]"#;
        let restored = restore_tool_names(output, &tool_defs);
        assert!(
            restored.contains("\"name\":\"getWeatherForecast\""),
            "longest match must win: {restored}"
        );
        assert!(
            !restored.contains("getWeather\","), // short match must not apply first
            "short match applied before long: {restored}"
        );
    }

    /// Regression: `max_enc - q_len - 1` previously underflowed (usize) when max_enc == 0.
    /// Verify the saturating version never panics across all edge cases.
    #[test]
    fn test_enc_input_truncation_no_underflow() {
        for max_enc in [0usize, 1, 2, 3, 512] {
            for query_len in [0usize, 1, 2, 100, 512, 1000] {
                let q_len = query_len.min(max_enc.saturating_sub(2));
                let _remaining = max_enc.saturating_sub(q_len + 1); // must not panic
                                                                    // Invariant: q_len + 1 + remaining <= max_enc (no overcounting)
                assert!(
                    q_len + 1 + _remaining <= max_enc || max_enc == 0,
                    "overcounting: q_len={q_len} remaining={_remaining} max_enc={max_enc}"
                );
            }
        }
    }

    /// Escaped quotes in description values must survive normalize_tools_json unchanged.
    #[test]
    fn test_normalize_tools_json_escaped_quote_in_description() {
        let json = r#"[{"name":"get_weather","description":"Get \"current\" weather"}]"#;
        let defs = ToolDef::from_json(json);
        let result = normalize_tools_json(json, &defs);
        // The description's escaped quotes must be preserved verbatim.
        assert!(
            result.contains(r#""Get \"current\" weather""#),
            "escaped quotes in description corrupted: {result}"
        );
        // The tool name must be untouched (already snake_case).
        assert!(
            result.contains(r#""name":"get_weather""#),
            "tool name was corrupted: {result}"
        );
    }

    /// Escaped quote inside a "name" value itself must not truncate the name.
    #[test]
    fn test_normalize_tools_json_name_with_escaped_quote() {
        // Unusual but valid JSON: a name containing a backslash-escaped quote.
        let json = r#"[{"name":"get\"weather","description":"test"}]"#;
        let defs = ToolDef::from_json(json);
        let result = normalize_tools_json(json, &defs);
        // The backslash-escaped quote inside the name value must not break scanning:
        // the name should be captured as `get\"weather` (7 chars before `"`) — or the
        // whole string — but critically the function must not panic.
        let _ = result; // just verify no panic
    }
}
