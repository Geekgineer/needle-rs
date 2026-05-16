//! Top-level Needle SAN (Simple Attention Network) model.
//!
//! Encoder: N layers of self-attention only (no FFN by default).
//! Decoder: M layers of self-attn + cross-attn (+ optional FFN).
//! Tied embeddings: embedding matrix is shared with LM head (transposed multiply).
//! Inference format:
//!   Encoder input: [query_tokens..., tools_id(5), tools_tokens...]
//!   Decoder start: [eos_id(1)]
//!   Decoder output: <tool_call>(4) then JSON

use alloc::vec::Vec;
use alloc::vec;
use crate::config::TransformerConfig;
use crate::attn::{AttnConfig, KvCache};
use crate::layers::{EncoderLayer, DecoderLayer, encoder_layer_forward, decoder_layer_forward};
use crate::rope::RopeCache;
use crate::norm::zc_rms_norm_vec;
use crate::math;

pub struct NeedleModel {
    pub cfg: TransformerConfig,

    /// Token embedding matrix [vocab_size, d_model] (also used as LM head transposed)
    pub embedding: Vec<f32>,

    pub encoder_layers: Vec<EncoderLayer>,
    pub decoder_layers: Vec<DecoderLayer>,

    /// Encoder final norm scale [d_model]
    pub encoder_final_norm: Vec<f32>,
    /// Decoder final norm scale [d_model]
    pub decoder_final_norm: Vec<f32>,

    /// RoPE cache (shared encoder + decoder)
    rope: RopeCache,

    attn_cfg: AttnConfig,

    /// sqrt(d_model) — applied to embeddings before encoder/decoder
    embed_scale: f32,
}

impl NeedleModel {
    pub fn new(
        cfg: TransformerConfig,
        embedding: Vec<f32>,
        encoder_layers: Vec<EncoderLayer>,
        decoder_layers: Vec<DecoderLayer>,
        encoder_final_norm: Vec<f32>,
        decoder_final_norm: Vec<f32>,
    ) -> Self {
        let rope = RopeCache::new(
            cfg.max_enc_len.max(cfg.max_dec_len),
            cfg.head_dim(),
            10000.0,
        );
        let attn_cfg = AttnConfig {
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim(),
            d_model: cfg.d_model,
        };
        let embed_scale = math::sqrt(cfg.d_model as f32);
        Self { cfg, embedding, encoder_layers, decoder_layers, encoder_final_norm, decoder_final_norm, rope, attn_cfg, embed_scale }
    }

    /// Embed a token and apply embedding scale: returns owned Vec.
    fn embed_scaled(&self, token_id: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        let idx = token_id as usize;
        debug_assert!(idx < self.cfg.vocab_size);
        let row = &self.embedding[idx * d..(idx + 1) * d];
        row.iter().map(|v| v * self.embed_scale).collect()
    }

    /// Compute logits over vocabulary for a hidden vector.
    /// Uses tied embedding (LM head = embedding^T).
    /// Dispatches to AVX2 path when compiled with `simd` feature on x86_64+avx2.
    fn lm_head(&self, hidden: &[f32], logits: &mut [f32]) {
        debug_assert_eq!(hidden.len(), self.cfg.d_model);
        debug_assert_eq!(logits.len(), self.cfg.vocab_size);

        #[cfg(all(target_arch = "x86_64", feature = "simd", target_feature = "avx2"))]
        // Safety: cfg guard ensures AVX2 is available at compile time.
        return unsafe { self.lm_head_avx2(hidden, logits) };

        #[allow(unreachable_code)]
        self.lm_head_scalar(hidden, logits);
    }

    fn lm_head_scalar(&self, hidden: &[f32], logits: &mut [f32]) {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        for tok in 0..v {
            let emb = &self.embedding[tok * d..(tok + 1) * d];
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += hidden[i] * emb[i];
            }
            logits[tok] = acc;
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn lm_head_avx2(&self, hidden: &[f32], logits: &mut [f32]) {
        use core::arch::x86_64::*;
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let chunks = d / 8;

        for tok in 0..v {
            let emb_ptr = self.embedding.as_ptr().add(tok * d);
            let mut acc = _mm256_setzero_ps();
            for c in 0..chunks {
                let h = _mm256_loadu_ps(hidden.as_ptr().add(c * 8));
                let e = _mm256_loadu_ps(emb_ptr.add(c * 8));
                acc = _mm256_fmadd_ps(h, e, acc);
            }
            // Horizontal sum: reduce 8 lanes → 1 scalar
            let hi128 = _mm256_extractf128_ps(acc, 1);
            let lo128 = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(hi128, lo128);
            let shuf = _mm_shuffle_ps(sum128, sum128, 0x4E);
            let sum64 = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_shuffle_ps(sum64, sum64, 0xB1);
            let sum32 = _mm_add_ss(sum64, shuf2);
            let mut result = _mm_cvtss_f32(sum32);
            // Scalar tail for d not divisible by 8
            for i in chunks * 8..d {
                result += hidden[i] * *emb_ptr.add(i);
            }
            logits[tok] = result;
        }
    }

    /// Run the encoder over the full input sequence.
    /// Returns encoder hidden states [enc_len, d_model].
    /// Also fills enc_kv_caches (one per decoder layer) with cross-attn K/V projections.
    pub fn encode(&self, input_ids: &[u32], enc_kv_caches: &mut Vec<KvCache>) -> Vec<f32> {
        let seq_len = input_ids.len();
        let d = self.cfg.d_model;

        // Embed + scale
        let mut x = vec![0.0f32; seq_len * d];
        for (t, &tok) in input_ids.iter().enumerate() {
            let row = &self.embedding[tok as usize * d..(tok as usize + 1) * d];
            for (dst, &src) in x[t * d..(t + 1) * d].iter_mut().zip(row.iter()) {
                *dst = src * self.embed_scale;
            }
        }

        let mut tmp = Vec::with_capacity(seq_len * d);
        let mut normed = vec![0.0f32; seq_len * d]; // pre-allocated scratch, reused across layers

        // Run encoder layers (self-attn only, no KV cache written here)
        for layer in self.encoder_layers.iter() {
            encoder_layer_forward(&mut x, layer, &self.attn_cfg, &self.rope, seq_len, None, &mut tmp, &mut normed);
        }

        // Encoder final norm
        for t in 0..seq_len {
            zc_rms_norm_vec(&mut x[t * d..(t + 1) * d], &self.encoder_final_norm);
        }

        // Project encoder output into cross-attn K/V for each decoder layer (no RoPE)
        self.fill_cross_kv(&x, seq_len, enc_kv_caches);

        x
    }

    /// Project encoder hidden states into K/V for each decoder layer's cross-attn.
    /// No RoPE applied (Python passes rope=None to cross-attention).
    fn fill_cross_kv(&self, enc_hidden: &[f32], seq_len: usize, kv_caches: &mut Vec<KvCache>) {
        let kv_h = self.cfg.num_kv_heads;
        let hd = self.cfg.head_dim();
        let kv_dim = kv_h * hd;

        for (li, layer) in self.decoder_layers.iter().enumerate() {
            let kv = &mut kv_caches[li];
            kv.reset();

            let mut k_all = vec![0.0f32; seq_len * kv_dim];
            let mut v_all = vec![0.0f32; seq_len * kv_dim];

            layer.cross_attn.wk.matmul(enc_hidden, seq_len, &mut k_all);
            layer.cross_attn.wv.matmul(enc_hidden, seq_len, &mut v_all);

            // Apply k_norm (ZCRMSNorm on each KV head slice)
            for t in 0..seq_len {
                for ki in 0..kv_h {
                    crate::norm::zc_rms_norm_vec(
                        &mut k_all[(t * kv_h + ki) * hd..(t * kv_h + ki + 1) * hd],
                        &layer.cross_attn.k_norm,
                    );
                }
            }

            // No RoPE on cross-attn keys (Python: rope=None)

            let kv_stride = kv_h * hd;
            for t in 0..seq_len {
                kv.push_kv(
                    &k_all[t * kv_stride..(t + 1) * kv_stride],
                    &v_all[t * kv_stride..(t + 1) * kv_stride],
                );
            }
        }
    }

    /// Autoregressive decode: generates one token given decoder input token and KV caches.
    /// Returns logits over the full vocabulary.
    pub fn decode_step(
        &self,
        token_id: u32,
        enc_kv_caches: &Vec<KvCache>,
        dec_kv_caches: &mut Vec<KvCache>,
        logits: &mut [f32],
    ) {
        let d = self.cfg.d_model;

        // Embed + scale
        let mut x: Vec<f32> = self.embed_scaled(token_id);
        let mut tmp = Vec::with_capacity(d);
        let mut normed = vec![0.0f32; d]; // pre-allocated scratch, reused across layers

        for (li, layer) in self.decoder_layers.iter().enumerate() {
            decoder_layer_forward(
                &mut x,
                layer,
                &self.attn_cfg,
                &self.rope,
                &enc_kv_caches[li],
                &mut dec_kv_caches[li],
                &mut tmp,
                &mut normed,
            );
        }

        // Decoder final norm + LM head
        zc_rms_norm_vec(&mut x, &self.decoder_final_norm);
        self.lm_head(&x, logits);
    }

    /// Allocate fresh KV caches for cross-attention (one per decoder layer, sized for enc input).
    pub fn make_enc_kv_caches(&self, enc_len: usize) -> Vec<KvCache> {
        (0..self.cfg.num_dec_layers)
            .map(|_| KvCache::new(enc_len, self.cfg.num_kv_heads, self.cfg.head_dim()))
            .collect()
    }

    /// Allocate fresh KV caches for decoder self-attention.
    pub fn make_dec_kv_caches(&self) -> Vec<KvCache> {
        (0..self.cfg.num_dec_layers)
            .map(|_| KvCache::new(self.cfg.max_dec_len, self.cfg.num_kv_heads, self.cfg.head_dim()))
            .collect()
    }
}
