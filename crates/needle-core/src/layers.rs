//! Encoder and decoder layer stacks.
//!
//! Gated residual: each sub-layer has a learned scalar sigmoid gate g (init=0).
//! output = x + sigmoid(g) * sublayer(norm(x))
//!
//! Encoder block: self-attn only (no FFN by default).
//! Decoder block: self-attn → cross-attn → optional FFN.

use alloc::vec::Vec;
use crate::attn::{AttnWeights, AttnConfig, KvCache, self_attn_full, self_attn_incremental, cross_attn_incremental};
use crate::ffn::{FfnWeights, ffn_forward};
use crate::norm::zc_rms_norm_vec;
use crate::rope::RopeCache;
use crate::ops::sigmoid;

pub struct EncoderLayer {
    pub self_attn: AttnWeights,
    pub self_attn_gate: f32,   // scalar sigmoid gate
    pub norm: Vec<f32>,        // ZCRMSNorm scale [d_model]
    pub ffn: Option<FfnWeights>,
    pub ffn_gate: f32,
    pub ffn_norm: Option<Vec<f32>>,
}

pub struct DecoderLayer {
    pub self_attn: AttnWeights,
    pub self_attn_gate: f32,
    pub self_attn_norm: Vec<f32>,
    pub cross_attn: AttnWeights,
    pub cross_attn_gate: f32,
    pub cross_attn_norm: Vec<f32>,
    pub ffn: Option<FfnWeights>,
    pub ffn_gate: f32,
    pub ffn_norm: Option<Vec<f32>>,
}

/// Run a single encoder layer on the full sequence.
/// `x`:      [seq_len, d_model] — modified in-place
/// `normed`: scratch buffer [seq_len, d_model] — pre-allocated by caller, reused across layers
pub fn encoder_layer_forward(
    x: &mut [f32],
    layer: &EncoderLayer,
    cfg: &AttnConfig,
    rope: &RopeCache,
    seq_len: usize,
    mask: Option<&[f32]>,
    tmp: &mut Vec<f32>,
    normed: &mut Vec<f32>,
) {
    let d = cfg.d_model;

    // Pre-norm (copy x into pre-allocated scratch, normalize in-place)
    normed.resize(seq_len * d, 0.0);
    normed.copy_from_slice(&x[..seq_len * d]);
    for t in 0..seq_len {
        zc_rms_norm_vec(&mut normed[t * d..(t + 1) * d], &layer.norm);
    }

    // Self-attention
    tmp.resize(seq_len * d, 0.0);
    self_attn_full(normed, tmp, &layer.self_attn, cfg, rope, seq_len, mask);

    // Gated residual
    let gate = sigmoid(layer.self_attn_gate);
    for i in 0..seq_len * d {
        x[i] += gate * tmp[i];
    }

    // Optional FFN
    if let (Some(ffn), Some(ffn_norm)) = (&layer.ffn, &layer.ffn_norm) {
        let d_ff = ffn.w1.out_feat;
        for t in 0..seq_len {
            let row = &mut x[t * d..(t + 1) * d];
            let mut normed_row = row.to_vec();
            zc_rms_norm_vec(&mut normed_row, ffn_norm);
            tmp.resize(d, 0.0);
            ffn_forward(&normed_row, tmp, ffn, d_ff);
            let ffn_gate = sigmoid(layer.ffn_gate);
            for (xi, &ti) in row.iter_mut().zip(tmp.iter()) {
                *xi += ffn_gate * ti;
            }
        }
    }
}

/// Run a single decoder layer incrementally (one token at a time).
/// `x`:      current decoder token vector [d_model] — modified in-place with residual
/// `enc_kv`: precomputed encoder KV (for cross-attn)
/// `dec_kv_self`: decoder self-attn KV cache (appended to here)
/// `normed`: scratch buffer [d_model] — pre-allocated by caller, reused for self-attn/cross-attn/ffn
pub fn decoder_layer_forward(
    x: &mut Vec<f32>,
    layer: &DecoderLayer,
    cfg: &AttnConfig,
    rope: &RopeCache,
    enc_kv: &KvCache,
    dec_kv_self: &mut KvCache,
    tmp: &mut Vec<f32>,
    normed: &mut Vec<f32>,
) {
    let d = cfg.d_model;

    // --- Self-attention ---
    normed.resize(d, 0.0);
    normed.copy_from_slice(x);
    zc_rms_norm_vec(normed, &layer.self_attn_norm);

    self_attn_incremental(normed, tmp, &layer.self_attn, cfg, rope, dec_kv_self, None);

    let gate = sigmoid(layer.self_attn_gate);
    for (xi, &ti) in x.iter_mut().zip(tmp.iter()) {
        *xi += gate * ti;
    }

    // --- Cross-attention (reuse normed buffer) ---
    normed.copy_from_slice(x);
    zc_rms_norm_vec(normed, &layer.cross_attn_norm);

    cross_attn_incremental(normed, tmp, &layer.cross_attn, cfg, enc_kv);

    let cross_gate = sigmoid(layer.cross_attn_gate);
    for (xi, &ti) in x.iter_mut().zip(tmp.iter()) {
        *xi += cross_gate * ti;
    }

    // --- Optional FFN (reuse normed buffer) ---
    if let (Some(ffn), Some(ffn_norm)) = (&layer.ffn, &layer.ffn_norm) {
        let d_ff = ffn.w1.out_feat;
        normed.copy_from_slice(x);
        zc_rms_norm_vec(normed, ffn_norm);
        tmp.resize(d, 0.0);
        ffn_forward(normed, tmp, ffn, d_ff);
        let ffn_gate = sigmoid(layer.ffn_gate);
        for (xi, &ti) in x.iter_mut().zip(tmp.iter()) {
            *xi += ffn_gate * ti;
        }
    }
}
