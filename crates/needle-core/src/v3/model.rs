//! Needle 3 model: weights and the whole-sequence forward pass.
//!
//! This is the prefill form — it computes every position at once, exactly as
//! upstream's training graph does, and is what parity is established against.
//! Incremental decode with a KV cache is layered on afterwards; getting the
//! order the other way round means debugging a cache against logits that were
//! never verified.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::cq::CqWeight;
use crate::kernels::rms_unit_to;
use crate::math::{cos, sin, sqrt};
use crate::norm::zc_rms_norm_vec;
use crate::ops::sigmoid;
use crate::v3::attention::{
    attend, attend_step, causal_depthwise_conv, norm_and_rope, AttnDims, Ring,
};
use crate::v3::cache::{Qkv, V3Cache};

/// Positions per batched-prefill chunk.
pub const DEFAULT_CHUNK: usize = 64;
use crate::v3::config::V3Config;
use crate::v3::engram::{engram_indices, ngram_valid, value_conv, EngramDims};
use crate::v3::kernels::{hadamard_mlp, HadaMlp, HadaPerms};
use crate::v3::mhc::{mix_down, scatter_up, Gates, LaneSite, MhcLayer};

/// One transformer layer's weights.
pub struct V3Layer {
    pub norm_in: Vec<f32>,
    pub q_proj: CqWeight,
    pub k_proj: CqWeight,
    pub v_proj: CqWeight,
    /// `(taps, dim)` each; empty when the model declares no conv.
    pub q_taps: Vec<f32>,
    pub k_taps: Vec<f32>,
    pub v_taps: Vec<f32>,
    /// Per-head-dim ZCRMSNorm scales, `qk_head_dim` long.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate_proj: CqWeight,
    pub out_proj: CqWeight,
    pub post_norm: Vec<f32>,
    /// Stored pre-sigmoid, as upstream stores it.
    pub attn_gate: f32,
    pub pre_hada: Vec<f32>,
    pub mlp: HadaMlp,
}

/// mHC parameters for the whole stack.
pub struct V3Mhc {
    /// `[num_layers]` each.
    pub a_pre: Vec<f32>,
    pub a_post: Vec<f32>,
    pub a_res: Vec<f32>,
    /// `[num_layers * lanes]`.
    pub b_pre: Vec<f32>,
    pub b_post: Vec<f32>,
    /// `[num_layers * lanes * lanes]`.
    pub b_res: Vec<f32>,
    /// `[num_layers * lanes, mhc_width]` — row `layer * lanes + lane`.
    pub phi_pre: CqWeight,
    pub phi_post: CqWeight,
    /// `[num_layers * lanes², mhc_width]`.
    pub phi_res: CqWeight,
}

/// One Engram site.
pub struct V3EngramSite {
    /// `[num_tables * slots, sub_dim]`.
    pub tables: CqWeight,
    pub key_proj: CqWeight,
    pub value_proj: CqWeight,
    /// `[conv_taps, d_model]`.
    pub taps: Vec<f32>,
}

/// A loaded Needle 3 model.
pub struct V3Model {
    pub cfg: V3Config,
    /// Tied embedding, `[vocab_size, d_model]` — also the LM head.
    pub embedding: CqWeight,
    pub layers: Vec<V3Layer>,
    pub mhc: V3Mhc,
    pub engrams: Vec<V3EngramSite>,
    pub final_norm: Vec<f32>,
    pub perms: HadaPerms,
    embed_scale: f32,
}

impl V3Model {
    pub fn new(
        cfg: V3Config,
        embedding: CqWeight,
        layers: Vec<V3Layer>,
        mhc: V3Mhc,
        engrams: Vec<V3EngramSite>,
        final_norm: Vec<f32>,
        perms: HadaPerms,
    ) -> Result<Self, &'static str> {
        if layers.len() != cfg.num_layers {
            return Err("layer count does not match geometry");
        }
        if engrams.len() != cfg.engram.sites.len() {
            return Err("engram site count does not match geometry");
        }
        if embedding.out_feat != cfg.vocab_size || embedding.in_feat != cfg.d_model {
            return Err("embedding shape does not match geometry");
        }
        let embed_scale = sqrt(cfg.d_model as f32);
        Ok(Self {
            cfg,
            embedding,
            layers,
            mhc,
            engrams,
            final_norm,
            perms,
            embed_scale,
        })
    }

    /// One incremental decode step: feed `token`, get logits for its position.
    ///
    /// Mathematically identical to [`Self::forward_sequence`] at the same
    /// position — the self-consistency test asserts exactly that.
    ///
    /// The mHC lane stream is deliberately **not** cached. Each position's
    /// lanes start from its own embedding and evolve using only that
    /// position's values; the sole cross-position dependencies are attention,
    /// the Q/K/V convolution and the Engram, and the cache carries each of
    /// those separately.
    pub fn decode_step(&self, cache: &mut V3Cache, token: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, n) = (cfg.d_model, cfg.mhc_lanes);
        let pos = cache.pos();
        let rows = cfg.logit_rows();

        cache.push_token(token);

        // ── Engram for this position ─────────────────────────────────────
        // The hash reads the last `max(order)` tokens; the cache keeps
        // exactly that many, oldest first, so the local index of "now" is the
        // end of that history.
        let e = &cfg.engram;
        let hist = cache.token_history().to_vec();
        let here = hist.len() - 1;
        let idx = engram_indices(&hist, &e.orders, e.heads, e.slots as u32, e.seed_heads);
        let num_tables = e.num_tables();

        let mut ek = vec![0.0f32; e.sites.len() * d];
        let mut ev = vec![0.0f32; e.sites.len() * d];
        let mut fetched = vec![0.0f32; num_tables * e.sub_dim];
        let mut row = vec![0.0f32; e.sub_dim];
        for (s, site) in self.engrams.iter().enumerate() {
            for t in 0..num_tables {
                let dst = &mut fetched[t * e.sub_dim..(t + 1) * e.sub_dim];
                // Validity is judged against the absolute position, not the
                // retained history length.
                if ngram_valid(pos, t, &e.orders, e.heads) {
                    let slot = idx[here * num_tables + t] as usize;
                    site.tables.dequantize_row(t * e.slots + slot, &mut row);
                    dst.copy_from_slice(&row);
                } else {
                    dst.fill(0.0);
                }
            }
            site.key_proj.matvec(&fetched, &mut ek[s * d..(s + 1) * d]);
            site.value_proj
                .matvec(&fetched, &mut ev[s * d..(s + 1) * d]);
            let mut cur = ev[s * d..(s + 1) * d].to_vec();
            cache.engram_conv_step(s, &site.taps, &mut cur);
            ev[s * d..(s + 1) * d].copy_from_slice(&cur);
        }

        // ── Lane stream for this position ────────────────────────────────
        let mut lanes = vec![0.0f32; n * d];
        let mut emb = vec![0.0f32; d];
        self.embedding.dequantize_row(token as usize, &mut emb);
        for x in emb.iter_mut() {
            *x *= self.embed_scale;
        }
        for lane in 0..n {
            lanes[lane * d..(lane + 1) * d].copy_from_slice(&emb);
        }

        let dims = AttnDims {
            seq: 1,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            qk_head_dim: cfg.qk_head_dim,
            v_head_dim: cfg.v_head_dim,
        };
        let half = cfg.qk_head_dim / 2;
        let (rc, rs) = self.rope_at(pos);

        let mut nx = vec![0.0f32; n * d];
        let mut nx_prep = vec![0.0f32; self.mhc.phi_pre.prepared_len()];
        let mut hpre = vec![0.0f32; n];
        let mut hpost = vec![0.0f32; n];
        let mut hres = vec![0.0f32; n * n];
        let mut scratch = vec![0.0f32; n * d];
        let mut u = vec![0.0f32; d];
        let mut q = vec![0.0f32; cfg.q_dim()];
        let mut k = vec![0.0f32; cfg.k_dim()];
        let mut v = vec![0.0f32; cfg.v_dim()];
        let mut attn = vec![0.0f32; cfg.attn_out_dim()];
        let mut gate = vec![0.0f32; cfg.attn_out_dim()];
        let mut proj = vec![0.0f32; d];
        let mut mlp_out = vec![0.0f32; d];

        for li in 0..cfg.num_layers {
            let layer = &self.layers[li];
            let m = MhcLayer {
                a_pre: self.mhc.a_pre[li],
                a_post: self.mhc.a_post[li],
                a_res: self.mhc.a_res[li],
                b_pre: &self.mhc.b_pre[li * n..(li + 1) * n],
                b_post: &self.mhc.b_post[li * n..(li + 1) * n],
                b_res: &self.mhc.b_res[li * n * n..(li + 1) * n * n],
            };

            rms_unit_to(&lanes, &mut nx);
            self.mhc.phi_pre.prepare_input(&nx, &mut nx_prep);
            self.mhc
                .phi_pre
                .matvec_rows_prepared(&nx_prep, li * n, &mut hpre);
            mix_down(&lanes, &mut hpre, &m, li, n, d, &mut u);

            let mut bx = u.clone();
            if let Some(site) = cfg.engram.site_of(li) {
                crate::v3::engram::apply_site(
                    &mut bx,
                    &ek[site * d..(site + 1) * d],
                    &ev[site * d..(site + 1) * d],
                    d,
                );
            }

            let mut h = bx.clone();
            zc_rms_norm_vec(&mut h, &layer.norm_in);
            layer.q_proj.matvec(&h, &mut q);
            layer.k_proj.matvec(&h, &mut k);
            layer.v_proj.matvec(&h, &mut v);

            if cfg.qkv_conv_taps > 0 {
                cache.conv_step(li, Qkv::Q, &layer.q_taps, &mut q);
                cache.conv_step(li, Qkv::K, &layer.k_taps, &mut k);
                cache.conv_step(li, Qkv::V, &layer.v_taps, &mut v);
            }
            norm_and_rope(
                &mut q,
                &layer.q_norm,
                &rc,
                &rs,
                1,
                cfg.num_heads,
                cfg.qk_head_dim,
            );
            norm_and_rope(
                &mut k,
                &layer.k_norm,
                &rc,
                &rs,
                1,
                cfg.num_kv_heads,
                cfg.qk_head_dim,
            );
            let _ = half;

            cache.write_kv(li, pos, &k, &v);
            let (lo, hi) = cache.span(li, pos);
            let slots = cache.slots(li);
            {
                let (kb, vb) = cache.kv(li);
                attend_step(
                    &q,
                    Ring {
                        k: kb,
                        v: vb,
                        slots,
                        lo,
                        hi,
                    },
                    dims,
                    &mut attn,
                );
            }

            layer.gate_proj.matvec(&h, &mut gate);
            for (ai, &gi) in attn.iter_mut().zip(gate.iter()) {
                *ai *= sigmoid(gi);
            }
            layer.out_proj.matvec(&attn, &mut proj);
            zc_rms_norm_vec(&mut proj, &layer.post_norm);
            let agate = sigmoid(layer.attn_gate);
            for (si, &pi) in bx.iter_mut().zip(proj.iter()) {
                *si += agate * pi;
            }

            proj.copy_from_slice(&bx);
            zc_rms_norm_vec(&mut proj, &layer.pre_hada);
            hadamard_mlp(&proj, &layer.mlp, &self.perms, cfg.hada_n, &mut mlp_out);
            for (si, &mi) in bx.iter_mut().zip(mlp_out.iter()) {
                *si += mi;
            }

            let mut y = vec![0.0f32; d];
            for c in 0..d {
                y[c] = bx[c] - u[c];
            }

            rms_unit_to(&lanes, &mut nx);
            self.mhc.phi_post.prepare_input(&nx, &mut nx_prep);
            self.mhc
                .phi_post
                .matvec_rows_prepared(&nx_prep, li * n, &mut hpost);
            self.mhc.phi_res.prepare_input(&nx, &mut nx_prep);
            self.mhc
                .phi_res
                .matvec_rows_prepared(&nx_prep, li * n * n, &mut hres);
            scatter_up(
                &mut lanes,
                Gates {
                    post: &mut hpost,
                    res: &mut hres,
                },
                &y,
                &m,
                LaneSite {
                    layer: li,
                    lanes: n,
                    d_model: d,
                },
                &mut scratch,
            );
        }

        cache.advance();

        let mut x = vec![0.0f32; d];
        for (c, xc) in x.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for lane in 0..n {
                acc += lanes[lane * d + c];
            }
            *xc = acc / n as f32;
        }
        zc_rms_norm_vec(&mut x, &self.final_norm);
        let mut lm_prep = vec![0.0f32; self.embedding.prepared_len()];
        let mut full = vec![0.0f32; cfg.vocab_size];
        self.embedding.prepare_input(&x, &mut lm_prep);
        self.embedding.matvec_prepared(&lm_prep, &mut full);
        full.truncate(rows);
        full
    }

    /// RoPE cos/sin for a single absolute position.
    fn rope_at(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.cfg.qk_head_dim / 2;
        let mut c = vec![0.0f32; half];
        let mut s = vec![0.0f32; half];
        for i in 0..half {
            let freq = 1.0
                / powf_f32(
                    self.cfg.rope_theta,
                    (2 * i) as f32 / self.cfg.qk_head_dim as f32,
                );
            let angle = pos as f32 * freq;
            c[i] = cos(angle);
            s[i] = sin(angle);
        }
        (c, s)
    }

    /// RoPE tables for `seq` positions: `(seq, qk_head_dim / 2)` each.
    fn rope(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.cfg.qk_head_dim / 2;
        let mut c = vec![0.0f32; seq * half];
        let mut s = vec![0.0f32; seq * half];
        for i in 0..half {
            let freq = 1.0
                / powf_f32(
                    self.cfg.rope_theta,
                    (2 * i) as f32 / self.cfg.qk_head_dim as f32,
                );
            for t in 0..seq {
                let angle = t as f32 * freq;
                c[t * half + i] = cos(angle);
                s[t * half + i] = sin(angle);
            }
        }
        (c, s)
    }

    /// Engram keys and values for every site, `(sites, seq, d_model)` each.
    ///
    /// With `want_raw`, also returns the **pre**-convolution values, which a
    /// cache needs to continue the dilated tap convolution from where a
    /// prefill left off.
    fn engram_kv_with_raw(&self, tokens: &[u32], want_raw: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let seq = tokens.len();
        let d = cfg.d_model;
        let e = &cfg.engram;
        let dims = EngramDims {
            num_tables: e.num_tables(),
            slots: e.slots,
            sub_dim: e.sub_dim,
            d_model: d,
            conv_taps: e.conv_taps,
            conv_dilation: e.conv_dilation,
            max_order: *e.orders.iter().max().unwrap_or(&1),
        };
        let indices = engram_indices(tokens, &e.orders, e.heads, e.slots as u32, e.seed_heads);

        let sites = e.sites.len();
        let mut ks = vec![0.0f32; sites * seq * d];
        let mut vs = vec![0.0f32; sites * seq * d];
        let mut raw = if want_raw {
            vec![0.0f32; sites * seq * d]
        } else {
            Vec::new()
        };
        let fetch_dim = dims.num_tables * dims.sub_dim;
        let mut fetched = vec![0.0f32; fetch_dim];
        let mut row = vec![0.0f32; dims.sub_dim];

        for (s, site) in self.engrams.iter().enumerate() {
            for p in 0..seq {
                for t in 0..dims.num_tables {
                    let dst = &mut fetched[t * dims.sub_dim..(t + 1) * dims.sub_dim];
                    if ngram_valid(p, t, &e.orders, e.heads) {
                        let slot = indices[p * dims.num_tables + t] as usize;
                        site.tables.dequantize_row(t * dims.slots + slot, &mut row);
                        dst.copy_from_slice(&row);
                    } else {
                        dst.fill(0.0);
                    }
                }
                let base = (s * seq + p) * d;
                site.key_proj.matvec(&fetched, &mut ks[base..base + d]);
                site.value_proj.matvec(&fetched, &mut vs[base..base + d]);
            }
            let off = s * seq * d;
            if want_raw {
                raw[off..off + seq * d].copy_from_slice(&vs[off..off + seq * d]);
            }
            value_conv(&mut vs[off..off + seq * d], &site.taps, seq, &dims);
        }
        (ks, vs, raw)
    }

    /// Logits for every position, `(seq, logit_rows)` row-major.
    ///
    /// Mirrors upstream's graph: the lane stream carries `mhc_lanes` copies of
    /// the scaled embedding, each layer mixes down and scatters back, and the
    /// head is the embedding itself, sliced to `out_vocab`.
    pub fn forward_sequence(&self, tokens: &[u32]) -> Vec<f32> {
        self.forward_impl(tokens, None)
    }

    /// Per-layer pooled states, `(seq, num_layers + 1, d_model)`.
    ///
    /// Cell 0 is the scaled input embedding; cell `1 + i` is the lane stream
    /// after layer `i`, meaned over lanes. These are what the probe heads read.
    pub fn forward_cells(&self, tokens: &[u32]) -> Vec<f32> {
        let mut cells = Vec::new();
        self.forward_impl(tokens, Some(&mut cells));
        cells
    }

    /// Batched prefill that also fills `cache`, so generation can continue
    /// from it with [`Self::decode_step`].
    ///
    /// Returns the logits for the **last** position, which is what a caller
    /// needs to sample the first new token. This is the path generation should
    /// use for a prompt: stepping the prompt through `decode_step` costs one
    /// full pass per position, where this pays the weight traffic once per
    /// chunk.
    pub fn prefill(&self, tokens: &[u32], cache: &mut V3Cache) -> Vec<f32> {
        let rows = self.cfg.logit_rows();
        let all = self.forward_impl_cached(tokens, None, Some(cache));
        let last = tokens.len() - 1;
        all[last * rows..(last + 1) * rows].to_vec()
    }

    fn forward_impl(&self, tokens: &[u32], cells: Option<&mut Vec<f32>>) -> Vec<f32> {
        self.forward_impl_cached(tokens, cells, None)
    }

    fn forward_impl_cached(
        &self,
        tokens: &[u32],
        mut cells: Option<&mut Vec<f32>>,
        mut cache: Option<&mut V3Cache>,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let (seq, d, n) = (tokens.len(), cfg.d_model, cfg.mhc_lanes);
        let rows = cfg.logit_rows();
        let (rc, rs) = self.rope(seq);
        let (ek, ev, ev_raw) = self.engram_kv_with_raw(tokens, cache.is_some());

        // Lane stream: (seq, lanes, d_model).
        let l1 = cfg.num_layers + 1;
        if let Some(c) = cells.as_deref_mut() {
            c.clear();
            c.resize(seq * l1 * d, 0.0);
        }
        let mut lanes = vec![0.0f32; seq * n * d];
        let mut emb = vec![0.0f32; d];
        for (t, &tok) in tokens.iter().enumerate() {
            self.embedding.dequantize_row(tok as usize, &mut emb);
            for e in emb.iter_mut() {
                *e *= self.embed_scale;
            }
            if let Some(c) = cells.as_deref_mut() {
                // Cell 0 is the scaled embedding, before the lane broadcast.
                c[(t * l1) * d..(t * l1 + 1) * d].copy_from_slice(&emb);
            }
            for lane in 0..n {
                let off = (t * n + lane) * d;
                lanes[off..off + d].copy_from_slice(&emb);
            }
        }

        let dims = AttnDims {
            seq,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            qk_head_dim: cfg.qk_head_dim,
            v_head_dim: cfg.v_head_dim,
        };
        let (q_dim, k_dim, v_dim) = (cfg.q_dim(), cfg.k_dim(), cfg.v_dim());
        let o_dim = cfg.attn_out_dim();

        let mut nx = vec![0.0f32; n * d];
        let mut nx_prep = vec![0.0f32; self.mhc.phi_pre.prepared_len()];
        let mut hpre = vec![0.0f32; n];
        let mut hpost = vec![0.0f32; n];
        let mut hres = vec![0.0f32; n * n];
        let mut scratch = vec![0.0f32; n * d];

        let mut u = vec![0.0f32; seq * d];
        let mut bx = vec![0.0f32; seq * d];
        let mut q = vec![0.0f32; seq * q_dim];
        let mut k = vec![0.0f32; seq * k_dim];
        let mut v = vec![0.0f32; seq * v_dim];
        let mut attn = vec![0.0f32; seq * o_dim];
        let mut gates = vec![0.0f32; seq * o_dim];

        // Chunked projection scratch. All four input projections share one
        // prepared activation, which is only sound because they share
        // `in_feat` and `group` — asserted rather than assumed.
        let chunk = DEFAULT_CHUNK.min(seq.max(1));
        let prep_len = self.layers[0].q_proj.prepared_len();
        debug_assert!(self.layers.iter().all(|l| {
            l.k_proj.prepared_len() == prep_len
                && l.v_proj.prepared_len() == prep_len
                && l.gate_proj.prepared_len() == prep_len
        }));
        let mut xh = vec![0.0f32; chunk * prep_len];
        let out_prep = self.layers[0].out_proj.prepared_len();
        let mut oh = vec![0.0f32; chunk * out_prep];
        let mut projected = vec![0.0f32; seq * d];
        let mut acc = vec![0.0f32; chunk];
        let mut mlp_out = vec![0.0f32; d];

        for li in 0..cfg.num_layers {
            let layer = &self.layers[li];
            let m = MhcLayer {
                a_pre: self.mhc.a_pre[li],
                a_post: self.mhc.a_post[li],
                a_res: self.mhc.a_res[li],
                b_pre: &self.mhc.b_pre[li * n..(li + 1) * n],
                b_post: &self.mhc.b_post[li * n..(li + 1) * n],
                b_res: &self.mhc.b_res[li * n * n..(li + 1) * n * n],
            };

            // Mix the lanes down to this layer's block input, per position.
            for t in 0..seq {
                let lane_slice = &lanes[t * n * d..(t + 1) * n * d];
                rms_unit_to(lane_slice, &mut nx);
                self.mhc.phi_pre.prepare_input(&nx, &mut nx_prep);
                self.mhc
                    .phi_pre
                    .matvec_rows_prepared(&nx_prep, li * n, &mut hpre);
                mix_down(
                    lane_slice,
                    &mut hpre,
                    &m,
                    li,
                    n,
                    d,
                    &mut u[t * d..(t + 1) * d],
                );
            }

            // Engram gate, if this layer carries a site.
            bx.copy_from_slice(&u);
            if let Some(site) = cfg.engram.site_of(li) {
                for t in 0..seq {
                    let base = (site * seq + t) * d;
                    crate::v3::engram::apply_site(
                        &mut bx[t * d..(t + 1) * d],
                        &ek[base..base + d],
                        &ev[base..base + d],
                        d,
                    );
                }
            }

            // ── Block ────────────────────────────────────────────────────
            // Pre-attention norm, then projections in chunks.
            //
            // A matvec streams the whole packed weight set once per position —
            // past any core's private cache — so a per-position loop re-reads
            // it every time. Batching decodes each group once and applies it to
            // the whole chunk, and `matmul_rows_prepared` is bit-identical to
            // repeated matvecs rather than merely close.
            //
            // q, k, v and the gate all read the same activation with the same
            // group geometry, so the Hadamard preparation is paid once for all
            // four rather than four times.
            let mut h = bx.clone();
            for t in 0..seq {
                zc_rms_norm_vec(&mut h[t * d..(t + 1) * d], &layer.norm_in);
            }
            for c0 in (0..seq).step_by(chunk) {
                let b = (seq - c0).min(chunk);
                for i in 0..b {
                    layer.q_proj.prepare_input(
                        &h[(c0 + i) * d..(c0 + i + 1) * d],
                        &mut xh[i * prep_len..(i + 1) * prep_len],
                    );
                }
                let xb = &xh[..b * prep_len];
                layer.q_proj.matmul_rows_prepared(
                    xb,
                    b,
                    0,
                    q_dim,
                    &mut q[c0 * q_dim..(c0 + b) * q_dim],
                    &mut acc,
                );
                layer.k_proj.matmul_rows_prepared(
                    xb,
                    b,
                    0,
                    k_dim,
                    &mut k[c0 * k_dim..(c0 + b) * k_dim],
                    &mut acc,
                );
                layer.v_proj.matmul_rows_prepared(
                    xb,
                    b,
                    0,
                    v_dim,
                    &mut v[c0 * v_dim..(c0 + b) * v_dim],
                    &mut acc,
                );
                layer.gate_proj.matmul_rows_prepared(
                    xb,
                    b,
                    0,
                    o_dim,
                    &mut gates[c0 * o_dim..(c0 + b) * o_dim],
                    &mut acc,
                );
            }

            // The conv runs in place, so the last few pre-conv vectors have to
            // be kept if the cache is going to continue from here.
            let (mut q_raw_tail, mut k_raw_tail, mut v_raw_tail) =
                (Vec::new(), Vec::new(), Vec::new());
            if cache.is_some() && cfg.qkv_conv_taps > 0 {
                let n = cfg.qkv_conv_taps - 1;
                q_raw_tail = last_rows(&q, seq, q_dim, n);
                k_raw_tail = last_rows(&k, seq, k_dim, n);
                v_raw_tail = last_rows(&v, seq, v_dim, n);
            }
            if cfg.qkv_conv_taps > 0 {
                causal_depthwise_conv(&mut q, &layer.q_taps, seq, q_dim, cfg.qkv_conv_taps);
                causal_depthwise_conv(&mut k, &layer.k_taps, seq, k_dim, cfg.qkv_conv_taps);
                causal_depthwise_conv(&mut v, &layer.v_taps, seq, v_dim, cfg.qkv_conv_taps);
            }
            norm_and_rope(
                &mut q,
                &layer.q_norm,
                &rc,
                &rs,
                seq,
                cfg.num_heads,
                cfg.qk_head_dim,
            );
            norm_and_rope(
                &mut k,
                &layer.k_norm,
                &rc,
                &rs,
                seq,
                cfg.num_kv_heads,
                cfg.qk_head_dim,
            );

            if let Some(c) = cache.as_deref_mut() {
                // Post-conv, post-norm, post-rope keys and values: exactly what
                // decode_step will attend against when it continues from here.
                for t in 0..seq {
                    c.write_kv(
                        li,
                        t,
                        &k[t * k_dim..(t + 1) * k_dim],
                        &v[t * v_dim..(t + 1) * v_dim],
                    );
                }
                if cfg.qkv_conv_taps > 0 {
                    c.seed_conv_tail(li, Qkv::Q, &q_raw_tail);
                    c.seed_conv_tail(li, Qkv::K, &k_raw_tail);
                    c.seed_conv_tail(li, Qkv::V, &v_raw_tail);
                }
            }

            attend(&q, &k, &v, dims, cfg.attention_span(li), &mut attn);

            let agate = sigmoid(layer.attn_gate);
            for t in 0..seq {
                let a = &mut attn[t * o_dim..(t + 1) * o_dim];
                for (ai, &gi) in a.iter_mut().zip(&gates[t * o_dim..(t + 1) * o_dim]) {
                    *ai *= sigmoid(gi);
                }
            }
            // out_proj reads the gated attention output, so it gets its own
            // preparation pass — a different activation and a different width
            // from the four input projections.
            for c0 in (0..seq).step_by(chunk) {
                let b = (seq - c0).min(chunk);
                for i in 0..b {
                    layer.out_proj.prepare_input(
                        &attn[(c0 + i) * o_dim..(c0 + i + 1) * o_dim],
                        &mut oh[i * out_prep..(i + 1) * out_prep],
                    );
                }
                layer.out_proj.matmul_rows_prepared(
                    &oh[..b * out_prep],
                    b,
                    0,
                    d,
                    &mut projected[c0 * d..(c0 + b) * d],
                    &mut acc,
                );
            }
            for t in 0..seq {
                let proj = &mut projected[t * d..(t + 1) * d];
                zc_rms_norm_vec(proj, &layer.post_norm);

                // Attention residual, then the MLP residual.
                let s1 = &mut bx[t * d..(t + 1) * d];
                for (si, &pi) in s1.iter_mut().zip(proj.iter()) {
                    *si += agate * pi;
                }
                proj.copy_from_slice(s1);
                zc_rms_norm_vec(proj, &layer.pre_hada);
                hadamard_mlp(proj, &layer.mlp, &self.perms, cfg.hada_n, &mut mlp_out);
                for (si, &mi) in s1.iter_mut().zip(mlp_out.iter()) {
                    *si += mi;
                }
            }

            // Scatter the block delta back across the lanes.
            for t in 0..seq {
                let lane_slice = &lanes[t * n * d..(t + 1) * n * d];
                rms_unit_to(lane_slice, &mut nx);
                self.mhc.phi_post.prepare_input(&nx, &mut nx_prep);
                self.mhc
                    .phi_post
                    .matvec_rows_prepared(&nx_prep, li * n, &mut hpost);
                self.mhc.phi_res.prepare_input(&nx, &mut nx_prep);
                self.mhc
                    .phi_res
                    .matvec_rows_prepared(&nx_prep, li * n * n, &mut hres);

                // y = block(u) - u
                let mut y = vec![0.0f32; d];
                for c in 0..d {
                    y[c] = bx[t * d + c] - u[t * d + c];
                }
                scatter_up(
                    &mut lanes[t * n * d..(t + 1) * n * d],
                    Gates {
                        post: &mut hpost,
                        res: &mut hres,
                    },
                    &y,
                    &m,
                    LaneSite {
                        layer: li,
                        lanes: n,
                        d_model: d,
                    },
                    &mut scratch,
                );
            }

            if let Some(c) = cells.as_deref_mut() {
                for t in 0..seq {
                    let dst = &mut c[(t * l1 + li + 1) * d..(t * l1 + li + 2) * d];
                    for (cc, o) in dst.iter_mut().enumerate() {
                        let mut a = 0.0f32;
                        for lane in 0..n {
                            a += lanes[(t * n + lane) * d + cc];
                        }
                        *o = a / n as f32;
                    }
                }
            }
        }

        if let Some(c) = cache {
            c.set_pos(seq);
            c.seed_tokens(tokens);
            let reach = cfg.engram.conv_taps.saturating_sub(1) * cfg.engram.conv_dilation;
            if reach > 0 {
                for s in 0..cfg.engram.sites.len() {
                    let off = s * seq * d;
                    c.seed_engram_tail(s, &last_rows(&ev_raw[off..off + seq * d], seq, d, reach));
                }
            }
        }

        // Mean over lanes, final norm, then the tied head — batched, because
        // at 8192 x 768 it is a fifth of the whole forward pass.
        let mut logits = vec![0.0f32; seq * rows];
        let mut pooled = vec![0.0f32; seq * d];
        for t in 0..seq {
            let out = &mut pooled[t * d..(t + 1) * d];
            for (c, o) in out.iter_mut().enumerate() {
                let mut a = 0.0f32;
                for lane in 0..n {
                    a += lanes[(t * n + lane) * d + c];
                }
                *o = a / n as f32;
            }
            zc_rms_norm_vec(out, &self.final_norm);
        }

        let lm_prep_len = self.embedding.prepared_len();
        let mut lh = vec![0.0f32; chunk * lm_prep_len];
        for c0 in (0..seq).step_by(chunk) {
            let b = (seq - c0).min(chunk);
            for i in 0..b {
                self.embedding.prepare_input(
                    &pooled[(c0 + i) * d..(c0 + i + 1) * d],
                    &mut lh[i * lm_prep_len..(i + 1) * lm_prep_len],
                );
            }
            // `rows` may be a prefix of the vocabulary: rows past `out_vocab`
            // are input-only code embeddings and are never scored.
            self.embedding.matmul_rows_prepared(
                &lh[..b * lm_prep_len],
                b,
                0,
                rows,
                &mut logits[c0 * rows..(c0 + b) * rows],
                &mut acc,
            );
        }
        logits
    }
}

/// The last `n` rows of a `(seq, dim)` buffer, oldest first, zero-padded at the
/// front when the sequence is shorter than `n`.
fn last_rows(buf: &[f32], seq: usize, dim: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * dim];
    for j in 0..n.min(seq) {
        // out is oldest-first, so the newest row lands last.
        let src = seq - n.min(seq) + j;
        let dst = n - n.min(seq) + j;
        out[dst * dim..(dst + 1) * dim].copy_from_slice(&buf[src * dim..(src + 1) * dim]);
    }
    out
}

/// `base^exp` for the RoPE frequency table. `crate::math::powf` exists but is
/// spelled differently across the `std` and `libm` backends.
fn powf_f32(base: f32, exp: f32) -> f32 {
    crate::math::powf(base, exp)
}
