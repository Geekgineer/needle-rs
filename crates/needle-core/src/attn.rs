//! Multi-head attention with GQA (Grouped Query Attention) and KV cache.
//!
//! Python model: 8 Q heads, 4 KV heads → kv_repeat = 2.
//! Each KV head serves 2 Q heads (broadcast, not repeat-copy in kernel).
//!
//! Projections per head:
//!   Q: [d_model, num_heads * head_dim]
//!   K: [d_model, num_kv_heads * head_dim]
//!   V: [d_model, num_kv_heads * head_dim]
//!   O: [num_heads * head_dim, d_model]
//!
//! Q and K are ZCRMSNorm'd before RoPE (matches Python MultiHeadAttention).

use crate::math;
use crate::norm::zc_rms_norm_vec;
use crate::ops::{dot, softmax_inplace};
use crate::quant::QuantizedWeight;
use crate::rope::RopeCache;
use alloc::vec;
use alloc::vec::Vec;

pub struct KvCache {
    /// Keys:   [max_len, num_kv_heads, head_dim]
    pub k: Vec<f32>,
    /// Values: [max_len, num_kv_heads, head_dim]
    pub v: Vec<f32>,
    pub len: usize,
    pub max_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new(max_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: vec![0.0; max_len * num_kv_heads * head_dim],
            v: vec![0.0; max_len * num_kv_heads * head_dim],
            len: 0,
            max_len,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    fn kv_stride(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn push_kv(&mut self, k_vec: &[f32], v_vec: &[f32]) {
        if self.len >= self.max_len {
            #[cfg(feature = "std")]
            eprintln!(
                "[needle-core] KV cache full at {} steps — output will degrade",
                self.max_len
            );
            return;
        }
        let stride = self.kv_stride();
        let base = self.len * stride;
        self.k[base..base + stride].copy_from_slice(k_vec);
        self.v[base..base + stride].copy_from_slice(v_vec);
        self.len += 1;
    }
}

/// Weights for a single MHA layer (shared between encoder self-attn, decoder self-attn, cross-attn).
pub struct AttnWeights {
    pub wq: QuantizedWeight, // [d_model, num_heads * head_dim]
    pub wk: QuantizedWeight, // [d_model, num_kv_heads * head_dim]
    pub wv: QuantizedWeight, // [d_model, num_kv_heads * head_dim]
    pub wo: QuantizedWeight, // [num_heads * head_dim, d_model]
    pub q_norm: Vec<f32>,    // ZCRMSNorm scale [head_dim]
    pub k_norm: Vec<f32>,    // ZCRMSNorm scale [head_dim]
}

pub struct AttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

impl AttnConfig {
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Self-attention forward pass for a single token (incremental decode with KV cache).
///
/// `x`:      input vector [d_model] — modified in-place to accumulate output
/// `out`:    output buffer [d_model]
/// `attn_mask`: optional additive mask; None = full causal (lower-triangular)
/// `cache`:  KV cache; len tokens already stored, this call appends position `cache.len`
#[allow(clippy::needless_range_loop)]
pub fn self_attn_incremental(
    x: &[f32],
    out: &mut Vec<f32>,
    weights: &AttnWeights,
    cfg: &AttnConfig,
    rope: &RopeCache,
    cache: &mut KvCache,
    attn_bias: Option<&[f32]>, // additive bias [cache.len+1]
) {
    let d = cfg.d_model;
    let h = cfg.num_heads;
    let kv_h = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let kv_dim = kv_h * hd;
    let q_dim = h * hd;

    // Project Q, K, V
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];

    weights.wq.matvec(x, &mut q);
    weights.wk.matvec(x, &mut k);
    weights.wv.matvec(x, &mut v);

    // ZCRMSNorm per head on Q and K
    for hi in 0..h {
        zc_rms_norm_vec(&mut q[hi * hd..(hi + 1) * hd], &weights.q_norm);
    }
    for ki in 0..kv_h {
        zc_rms_norm_vec(&mut k[ki * hd..(ki + 1) * hd], &weights.k_norm);
    }

    // RoPE on Q and K (single token, offset = cache.len)
    rope.apply(&mut q, 1, h, hd, cache.len);
    rope.apply(&mut k, 1, kv_h, hd, cache.len);

    // Append to KV cache
    cache.push_kv(&k, &v);
    let ctx_len = cache.len; // now includes current token

    // Scaled dot-product attention, one head at a time
    let scale = math::powf(hd as f32, -0.5);
    let mut attn_scores = vec![0.0f32; ctx_len];
    let kv_stride = kv_h * hd;

    out.clear();
    out.resize(d, 0.0);
    let mut head_out = vec![0.0f32; d]; // accumulate O-proj input

    for hi in 0..h {
        let kv_hi = hi / cfg.kv_repeat(); // GQA: which KV head serves this Q head
        let q_head = &q[hi * hd..(hi + 1) * hd];

        // Compute attention scores for all cached positions
        for t in 0..ctx_len {
            let k_head = &cache.k[t * kv_stride + kv_hi * hd..t * kv_stride + (kv_hi + 1) * hd];
            let mut score = dot(q_head, k_head) * scale;
            if let Some(bias) = attn_bias {
                score += bias[t];
            }
            attn_scores[t] = score;
        }

        // Causal mask: last token attends to [0..ctx_len], which is exactly what's cached — no masking needed
        softmax_inplace(&mut attn_scores[..ctx_len]);

        // Weighted sum of V
        let out_head = &mut head_out[hi * hd..(hi + 1) * hd];
        for v_el in out_head.iter_mut() {
            *v_el = 0.0;
        }
        for t in 0..ctx_len {
            let v_head = &cache.v[t * kv_stride + kv_hi * hd..t * kv_stride + (kv_hi + 1) * hd];
            let a = attn_scores[t];
            for (o, &vi) in out_head.iter_mut().zip(v_head.iter()) {
                *o += a * vi;
            }
        }
    }

    // Output projection: out = head_out @ Wo
    weights.wo.matvec(&head_out, out);
}

/// Full-sequence self-attention (encoder, no KV cache, no causal mask by default).
/// `x`:     [seq_len, d_model] input, modified in-place
/// `out`:   [seq_len, d_model] output
/// `mask`:  optional additive bias [seq_len, seq_len]  (−inf for masked positions)
pub fn self_attn_full(
    x: &[f32],
    out: &mut [f32],
    weights: &AttnWeights,
    cfg: &AttnConfig,
    rope: &RopeCache,
    seq_len: usize,
    mask: Option<&[f32]>,
) {
    let d = cfg.d_model;
    let h = cfg.num_heads;
    let kv_h = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let q_dim = h * hd;
    let kv_dim = kv_h * hd;

    debug_assert_eq!(x.len(), seq_len * d);
    debug_assert_eq!(out.len(), seq_len * d);

    let mut q = vec![0.0f32; seq_len * q_dim];
    let mut k = vec![0.0f32; seq_len * kv_dim];
    let mut v = vec![0.0f32; seq_len * kv_dim];

    // Project all tokens
    weights.wq.matmul(x, seq_len, &mut q);
    weights.wk.matmul(x, seq_len, &mut k);
    weights.wv.matmul(x, seq_len, &mut v);

    // ZCRMSNorm per head on Q and K
    for t in 0..seq_len {
        for hi in 0..h {
            zc_rms_norm_vec(
                &mut q[(t * h + hi) * hd..(t * h + hi + 1) * hd],
                &weights.q_norm,
            );
        }
        for ki in 0..kv_h {
            zc_rms_norm_vec(
                &mut k[(t * kv_h + ki) * hd..(t * kv_h + ki + 1) * hd],
                &weights.k_norm,
            );
        }
    }

    // RoPE (no offset for encoder)
    rope.apply(&mut q, seq_len, h, hd, 0);
    rope.apply(&mut k, seq_len, kv_h, hd, 0);

    let scale = math::powf(hd as f32, -0.5);
    let mut head_out = vec![0.0f32; seq_len * q_dim];
    let mut scores = vec![0.0f32; seq_len];

    for t in 0..seq_len {
        for hi in 0..h {
            let kv_hi = hi / cfg.kv_repeat();
            let q_head = &q[(t * h + hi) * hd..(t * h + hi + 1) * hd];

            for s in 0..seq_len {
                let k_head = &k[(s * kv_h + kv_hi) * hd..(s * kv_h + kv_hi + 1) * hd];
                let mut score = dot(q_head, k_head) * scale;
                if let Some(m) = mask {
                    score += m[t * seq_len + s];
                }
                scores[s] = score;
            }

            softmax_inplace(&mut scores[..seq_len]);

            let out_head = &mut head_out[(t * h + hi) * hd..(t * h + hi + 1) * hd];
            for o in out_head.iter_mut() {
                *o = 0.0;
            }
            for s in 0..seq_len {
                let v_head = &v[(s * kv_h + kv_hi) * hd..(s * kv_h + kv_hi + 1) * hd];
                let a = scores[s];
                for (o, &vi) in out_head.iter_mut().zip(v_head.iter()) {
                    *o += a * vi;
                }
            }
        }
    }

    // O projection
    weights.wo.matmul(&head_out, seq_len, out);
}

/// Cross-attention: queries from decoder, keys/values from encoder memory.
/// `q_in`:   [1, d_model] decoder query (single step)
/// `mem`:    [enc_len, d_model] encoder output
/// `out`:    [d_model] output vector
/// Uses precomputed encoder K/V (stored in a separate KvCache filled from encoder run).
#[allow(clippy::needless_range_loop)]
pub fn cross_attn_incremental(
    q_in: &[f32],
    out: &mut Vec<f32>,
    weights: &AttnWeights,
    cfg: &AttnConfig,
    enc_kv: &KvCache, // precomputed cross-attn K/V from encoder hidden states; no RoPE (Python: rope=None)
) {
    let d = cfg.d_model;
    let h = cfg.num_heads;
    let kv_h = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let q_dim = h * hd;
    let ctx_len = enc_kv.len;

    let mut q = vec![0.0f32; q_dim];
    weights.wq.matvec(q_in, &mut q);

    for hi in 0..h {
        zc_rms_norm_vec(&mut q[hi * hd..(hi + 1) * hd], &weights.q_norm);
    }

    // Python passes rope=None for cross-attention, so neither Q nor K are RoPE'd here.
    // Encoder K already has RoPE applied from the encoder forward pass.

    let scale = math::powf(hd as f32, -0.5);
    let kv_stride = kv_h * hd;
    let mut scores = vec![0.0f32; ctx_len];
    let mut head_out = vec![0.0f32; q_dim];

    for hi in 0..h {
        let kv_hi = hi / cfg.kv_repeat();
        let q_head = &q[hi * hd..(hi + 1) * hd];

        for t in 0..ctx_len {
            let k_head = &enc_kv.k[t * kv_stride + kv_hi * hd..t * kv_stride + (kv_hi + 1) * hd];
            scores[t] = dot(q_head, k_head) * scale;
        }

        softmax_inplace(&mut scores[..ctx_len]);

        let out_head = &mut head_out[hi * hd..(hi + 1) * hd];
        for o in out_head.iter_mut() {
            *o = 0.0;
        }
        for t in 0..ctx_len {
            let v_head = &enc_kv.v[t * kv_stride + kv_hi * hd..t * kv_stride + (kv_hi + 1) * hd];
            let a = scores[t];
            for (o, &vi) in out_head.iter_mut().zip(v_head.iter()) {
                *o += a * vi;
            }
        }
    }

    out.clear();
    out.resize(d, 0.0);
    weights.wo.matvec(&head_out, out);
}
