//! High-level Needle inference engine.
//!
//! Orchestrates: SafeTensors load → model construction → encode → greedy decode.

use std::path::Path;
use needle_core::{
    TransformerConfig,
    model::NeedleModel,
    attn::KvCache,
    layers::{EncoderLayer, DecoderLayer},
    quant::QuantizedWeight,
};
use crate::safetensors::SafeTensors;
use crate::tokenizer::{Vocabulary, to_snake_case, EOS_ID, TOOL_CALL_ID, TOOLS_ID};
use crate::constrained::{ConstrainedDecoder, ConstraintState, ToolDef};

pub struct InferenceResult {
    pub token_ids: Vec<u32>,
    pub text: String,
}

pub struct NeedleEngine {
    model: NeedleModel,
    vocab: Vocabulary,
    max_gen_len: usize,
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

        let cfg = load_config_from_safetensors(&st);
        let model = load_model(&st, &cfg);

        Ok(Self {
            model,
            vocab,
            max_gen_len: cfg.max_dec_len,
        })
    }

    /// Run single-example inference.
    /// `query`:      raw query string (tokenized internally)
    /// `tools_json`: JSON string of tool definitions array (tokenized internally)
    pub fn run(&self, query: &str, tools_json: &str) -> InferenceResult {
        // Tokenize: encode(query) + [TOOLS_ID] + encode(tools_json)
        let query_ids = self.vocab.encode(query);
        let tools_ids = self.vocab.encode(tools_json);

        let mut enc_input = Vec::with_capacity(query_ids.len() + 1 + tools_ids.len());
        enc_input.extend_from_slice(&query_ids);
        enc_input.push(TOOLS_ID as u32);
        enc_input.extend_from_slice(&tools_ids);

        // Truncate to max_enc_len
        let max_enc = self.model.cfg.max_enc_len;
        enc_input.truncate(max_enc);

        // Allocate KV caches
        let enc_len = enc_input.len();
        let mut enc_kv = self.model.make_enc_kv_caches(enc_len);
        let mut dec_kv = self.model.make_dec_kv_caches();

        // Encode
        self.model.encode(&enc_input, &mut enc_kv);

        // Set up constrained decoder
        let tool_defs = ToolDef::from_json(tools_json);
        // Build token byte map for trie lookups
        let token_bytes: Vec<(u32, Vec<u8>)> = self.vocab.id_to_piece.iter()
            .enumerate()
            .map(|(i, piece)| {
                let bytes = piece.replace('▁', " ").into_bytes();
                (i as u32, bytes)
            })
            .collect();
        let constrained = ConstrainedDecoder::new(&tool_defs, token_bytes);

        // Greedy decode starting from [EOS]
        let mut output_ids = Vec::with_capacity(64);
        let mut current_token = EOS_ID;
        let mut logits = vec![0.0f32; self.model.cfg.vocab_size];
        let mut constraint_state = ConstraintState::Free;
        // Track output bytes for constraint state transitions
        let mut output_text = String::new();

        for _step in 0..self.max_gen_len {
            self.model.decode_step(current_token, &enc_kv, &mut dec_kv, &mut logits);

            // Apply constraint mask
            let mask = constrained.logit_mask(&constraint_state, self.model.cfg.vocab_size);
            for (l, &m) in logits.iter_mut().zip(mask.iter()) {
                *l += m;
            }

            // Greedy argmax
            let next_token = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(EOS_ID);

            if next_token == EOS_ID {
                break;
            }

            output_ids.push(next_token);
            current_token = next_token;

            // Update output text and constraint state
            if let Some(piece) = self.vocab.id_to_piece.get(next_token as usize) {
                output_text.push_str(&piece.replace('▁', " "));
            }

            // Transition constraint state based on output_text content
            constraint_state = update_constraint_state(&output_text, &constraint_state);
        }

        let text = self.vocab.decode_ids(&output_ids);

        InferenceResult { token_ids: output_ids, text }
    }
}

/// Update constraint state machine based on accumulated output text.
fn update_constraint_state(text: &str, current: &ConstraintState) -> ConstraintState {
    // After seeing `"name":"` → enter ToolName constraint
    if text.contains("\"name\":\"") && *current == ConstraintState::Free {
        return ConstraintState::ToolName { trie_node: 0 };
    }
    // After seeing closing `"` on tool name → back to Free (then argument keys)
    // Simplified: use Free for now, proper state tracking added in next iteration
    current.clone()
}

/// Extract model config from SafeTensors metadata (or use defaults).
fn load_config_from_safetensors(st: &SafeTensors) -> TransformerConfig {
    // Config is stored as __metadata__ in the SafeTensors file by the export script.
    // Fall back to defaults if not found.
    TransformerConfig::default()
}

/// Build NeedleModel from loaded SafeTensors tensors.
/// Tensor naming convention (set by export.py):
///   embedding:                   "embedding"
///   encoder layer i self-attn:   "encoder.{i}.self_attn.wq", ...
///   decoder layer i self-attn:   "decoder.{i}.self_attn.wq", ...
///   decoder layer i cross-attn:  "decoder.{i}.cross_attn.wq", ...
///   norms:                       "encoder.{i}.norm", "decoder.{i}.self_attn_norm", etc.
///   gates:                       "encoder.{i}.self_attn_gate", etc.
fn load_model(st: &SafeTensors, cfg: &TransformerConfig) -> NeedleModel {
    let d = cfg.d_model;
    let v = cfg.vocab_size;

    let embedding = st.get_f32("embedding")
        .expect("missing embedding tensor");
    assert_eq!(embedding.len(), v * d);

    let encoder_layers = (0..cfg.num_layers)
        .map(|i| load_encoder_layer(st, cfg, i))
        .collect();

    let decoder_layers = (0..cfg.num_dec_layers)
        .map(|i| load_decoder_layer(st, cfg, i))
        .collect();

    let encoder_final_norm = st.get_f32("encoder_final_norm")
        .unwrap_or_else(|| vec![0.0f32; d]);
    let decoder_final_norm = st.get_f32("decoder_final_norm")
        .unwrap_or_else(|| vec![0.0f32; d]);

    NeedleModel::new(cfg.clone(), embedding, encoder_layers, decoder_layers, encoder_final_norm, decoder_final_norm)
}

fn load_encoder_layer(st: &SafeTensors, cfg: &TransformerConfig, i: usize) -> EncoderLayer {
    let prefix = format!("encoder.{i}");
    EncoderLayer {
        self_attn: load_attn_weights(st, cfg, &format!("{prefix}.self_attn")),
        self_attn_gate: load_scalar(st, &format!("{prefix}.self_attn_gate")),
        norm: load_vec(st, &format!("{prefix}.norm"), cfg.d_model),
        ffn: None,  // no_feedforward=true by default
        ffn_gate: 0.0,
        ffn_norm: None,
    }
}

fn load_decoder_layer(st: &SafeTensors, cfg: &TransformerConfig, i: usize) -> DecoderLayer {
    let prefix = format!("decoder.{i}");
    DecoderLayer {
        self_attn: load_attn_weights(st, cfg, &format!("{prefix}.self_attn")),
        self_attn_gate: load_scalar(st, &format!("{prefix}.self_attn_gate")),
        self_attn_norm: load_vec(st, &format!("{prefix}.self_attn_norm"), cfg.d_model),
        cross_attn: load_attn_weights(st, cfg, &format!("{prefix}.cross_attn")),
        cross_attn_gate: load_scalar(st, &format!("{prefix}.cross_attn_gate")),
        cross_attn_norm: load_vec(st, &format!("{prefix}.cross_attn_norm"), cfg.d_model),
        ffn: None,
        ffn_gate: 0.0,
        ffn_norm: None,
    }
}

fn load_attn_weights(st: &SafeTensors, cfg: &TransformerConfig, prefix: &str) -> needle_core::attn::AttnWeights {
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
    st.get_f32(name).and_then(|v| v.first().copied()).unwrap_or(0.0)
}
