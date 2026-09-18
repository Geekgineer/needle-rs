//! Incremental decode state for Needle 3.
//!
//! v3 carries three separate histories, not one, and each has a different
//! reach. Missing any of them produces output that looks plausible and is
//! wrong, so they are modelled explicitly:
//!
//! 1. **Keys and values**, per layer. Sixteen layers attend over a 1024
//!    window and get a fixed ring; four attend over everything and grow.
//!    This is the only place v2's single-ring design does not transfer.
//! 2. **Raw Q/K/V projections**, the last `qkv_conv_taps - 1` of them per
//!    layer, because the causal conv over the sequence needs them to form the
//!    current position's post-conv vectors.
//! 3. **Engram history** — the last few token ids for the n-gram hash, and the
//!    pre-convolution value vectors reaching back `(conv_taps - 1) · dilation`
//!    positions per site.
//!
//! Memory is sized to the session, not to the windows. Every layer starts at
//! the caller's hint and doubles into its cap — a local layer stops at its
//! 1024 window and then wraps as a ring, a global layer keeps growing. While a
//! ring is shorter than its cap the modulo is the identity, so growing can
//! never scramble what is already written.
//!
//! That is worth roughly 4x on a typical session: reserving every window and
//! the full context costs 41 MB at f32, where a 512-token session pays 8.8 MB.
//! The container declares `kv_bits = 8`, so upstream post-trained this model
//! for an int8 cache — another 4x — but quantising here would change numerics
//! away from the verified path, so it is tracked as separate work rather than
//! done silently.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::v3::config::V3Config;

/// Per-layer key/value storage.
struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Ring capacity in positions, grown on demand up to `cap`.
    slots: usize,
    /// The most this layer will ever hold: its window, or `max_seq_len` for a
    /// global layer.
    cap: usize,
    /// Raw, pre-convolution projections for the conv tail, most recent last.
    /// `(taps - 1)` entries of `q_dim`, `k_dim`, `v_dim`.
    q_tail: Vec<f32>,
    k_tail: Vec<f32>,
    v_tail: Vec<f32>,
}

/// Decode state for one session.
pub struct V3Cache {
    cfg: V3Config,
    layers: Vec<LayerCache>,
    /// Token ids, kept only as far back as the largest n-gram order needs.
    tokens: Vec<u32>,
    /// Pre-convolution Engram values per site, most recent last, reaching
    /// `(conv_taps - 1) * dilation` positions.
    engram_tail: Vec<Vec<f32>>,
    /// Number of positions written so far; the next write lands here.
    pos: usize,
}

impl V3Cache {
    /// A cache for a session expected to reach `hint` positions.
    ///
    /// `hint` only sizes the first allocation for global layers — exceeding it
    /// grows rather than fails. Local layers are always exactly their window.
    pub fn new(cfg: &V3Config, hint: usize) -> Self {
        let hint = hint.clamp(1, cfg.max_seq_len);
        let taps_tail = cfg.qkv_conv_taps.saturating_sub(1);
        let layers = (0..cfg.num_layers)
            .map(|li| {
                // Grow into the window rather than reserving it. While the
                // ring is still shorter than its cap the modulo is the
                // identity, so growing can never scramble what is already
                // written; once it reaches the cap it wraps as a ring.
                let cap = match cfg.attention_span(li) {
                    Some(w) => w.min(cfg.max_seq_len),
                    None => cfg.max_seq_len,
                };
                let slots = hint.min(cap);
                LayerCache {
                    k: vec![0.0; slots * cfg.k_dim()],
                    v: vec![0.0; slots * cfg.v_dim()],
                    slots,
                    cap,
                    q_tail: vec![0.0; taps_tail * cfg.q_dim()],
                    k_tail: vec![0.0; taps_tail * cfg.k_dim()],
                    v_tail: vec![0.0; taps_tail * cfg.v_dim()],
                }
            })
            .collect();

        let reach = cfg.engram.conv_taps.saturating_sub(1) * cfg.engram.conv_dilation;
        let engram_tail = (0..cfg.engram.sites.len())
            .map(|_| vec![0.0; reach * cfg.d_model])
            .collect();

        Self {
            cfg: cfg.clone(),
            layers,
            tokens: Vec::with_capacity(hint),
            engram_tail,
            pos: 0,
        }
    }

    /// Positions written so far.
    pub fn len(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Bytes currently held, so a caller on a constrained target can see what
    /// a session actually costs rather than guessing.
    pub fn bytes(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| (l.k.len() + l.v.len() + l.q_tail.len() + l.k_tail.len() + l.v_tail.len()) * 4)
            .sum();
        let engram: usize = self.engram_tail.iter().map(|t| t.len() * 4).sum();
        per_layer + engram + self.tokens.len() * 4
    }

    /// Token ids retained for the n-gram hash, oldest first.
    pub fn token_history(&self) -> &[u32] {
        &self.tokens
    }

    /// Record a token, dropping history no longer reachable by any hash.
    pub fn push_token(&mut self, tok: u32) {
        let keep = self.cfg.engram.orders.iter().copied().max().unwrap_or(1);
        self.tokens.push(tok);
        if self.tokens.len() > keep {
            let drop = self.tokens.len() - keep;
            self.tokens.drain(..drop);
        }
    }

    /// Grow layer `li` so `pos` is addressable, up to its cap.
    ///
    /// A local layer stops at its window and then wraps; a global layer keeps
    /// growing to `max_seq_len`. Doubling keeps the amortised cost linear
    /// without ever reserving the maximum up front — which is the difference
    /// between a 2 MB session and a 14 MB one.
    fn ensure(&mut self, li: usize, pos: usize) {
        let (k_dim, v_dim) = (self.cfg.k_dim(), self.cfg.v_dim());
        let layer = &mut self.layers[li];
        if pos < layer.slots || layer.slots >= layer.cap {
            return;
        }
        let want = (layer.slots * 2).max(pos + 1).min(layer.cap);
        layer.k.resize(want * k_dim, 0.0);
        layer.v.resize(want * v_dim, 0.0);
        layer.slots = want;
    }

    /// The inclusive logical range layer `li` may attend to at `pos`.
    ///
    /// Clamped to the ring's capacity. A global layer past `max_seq_len` would
    /// otherwise ask for more history than it stores, and because slots are
    /// addressed modulo the capacity it would read *itself* as its own past —
    /// silently, since the bounds check is a debug assertion. Degrading to the
    /// most recent `slots` positions is wrong too, but it is bounded, visible
    /// through [`Self::is_saturated`], and cannot corrupt.
    ///
    /// Callers should avoid reaching this: see `V3Engine`, which bounds the
    /// prompt and reports `StopReason::MaxSeqLen`.
    pub fn span(&self, li: usize, pos: usize) -> (usize, usize) {
        let slots = self.layers[li].slots;
        let lo = match self.cfg.attention_span(li) {
            Some(w) if pos + 1 > w => pos + 1 - w,
            _ => 0,
        };
        // Never ask for a wider window than the ring holds.
        let lo = lo.max((pos + 1).saturating_sub(slots));
        (lo, pos)
    }

    /// True once any layer has run past what it can store, so attention is no
    /// longer seeing the history the model was trained to see.
    pub fn is_saturated(&self) -> bool {
        (0..self.cfg.num_layers)
            .any(|li| self.cfg.attention_span(li).is_none() && self.pos > self.layers[li].slots)
    }

    /// Ring capacity for layer `li`.
    pub fn slots(&self, li: usize) -> usize {
        self.layers[li].slots
    }

    /// Write this position's post-convolution key and value.
    pub fn write_kv(&mut self, li: usize, pos: usize, k: &[f32], v: &[f32]) {
        self.ensure(li, pos);
        let (k_dim, v_dim) = (self.cfg.k_dim(), self.cfg.v_dim());
        let layer = &mut self.layers[li];
        let slot = pos % layer.slots;
        layer.k[slot * k_dim..(slot + 1) * k_dim].copy_from_slice(k);
        layer.v[slot * v_dim..(slot + 1) * v_dim].copy_from_slice(v);
    }

    /// Key and value buffers for layer `li`.
    pub fn kv(&self, li: usize) -> (&[f32], &[f32]) {
        (&self.layers[li].k, &self.layers[li].v)
    }

    /// Apply the causal conv for one position using the retained raw tail,
    /// then roll the tail forward.
    ///
    /// `raw` is this position's pre-conv projection and is replaced with the
    /// post-conv result. History before the start of the session contributes
    /// zero, exactly as the reference's zero-padded shift does.
    pub fn conv_step(&mut self, li: usize, which: Qkv, taps: &[f32], raw: &mut [f32]) {
        let n_taps = self.cfg.qkv_conv_taps;
        if n_taps == 0 {
            return;
        }
        let dim = raw.len();
        let pos = self.pos;
        let layer = &mut self.layers[li];
        let tail = match which {
            Qkv::Q => &mut layer.q_tail,
            Qkv::K => &mut layer.k_tail,
            Qkv::V => &mut layer.v_tail,
        };
        debug_assert_eq!(tail.len(), (n_taps - 1) * dim);

        let mut out = vec![0.0f32; dim];
        for c in 0..dim {
            let mut acc = taps[c] * raw[c];
            for j in 1..n_taps {
                if j > pos {
                    break;
                }
                // tail is most-recent-last, so position pos-j is at index
                // (n_taps - 1 - j).
                let slot = n_taps - 1 - j;
                acc += taps[j * dim + c] * tail[slot * dim + c];
            }
            out[c] = acc;
        }

        // Roll: drop the oldest, append this position's raw vector.
        if n_taps > 1 {
            tail.copy_within(dim.., 0);
            let last = (n_taps - 2) * dim;
            tail[last..last + dim].copy_from_slice(raw);
        }
        raw.copy_from_slice(&out);
    }

    /// The Engram value convolution for one position, using the retained tail.
    ///
    /// `raw` is this position's pre-conv Engram value and is replaced with the
    /// post-conv result.
    pub fn engram_conv_step(&mut self, site: usize, taps: &[f32], raw: &mut [f32]) {
        let e = &self.cfg.engram;
        let (n_taps, dil) = (e.conv_taps, e.conv_dilation);
        let max_order = e.orders.iter().copied().max().unwrap_or(1);
        let d = self.cfg.d_model;
        let pos = self.pos;
        let reach = (n_taps - 1) * dil;
        let tail = &mut self.engram_tail[site];

        let mut out = vec![0.0f32; d];
        for c in 0..d {
            let mut acc = 0.0f32;
            for j in 0..n_taps {
                let back = j * dil;
                if back > pos || pos < j * max_order {
                    continue;
                }
                let val = if j == 0 {
                    raw[c]
                } else {
                    // tail holds the last `reach` positions, most recent last,
                    // so position pos-back sits at index reach-back.
                    tail[(reach - back) * d + c]
                };
                acc += taps[j * d + c] * val;
            }
            out[c] = acc;
        }

        if reach > 0 {
            tail.copy_within(d.., 0);
            let last = (reach - 1) * d;
            tail[last..last + d].copy_from_slice(raw);
        }
        raw.copy_from_slice(&out);
    }

    /// Seed the conv tail for a layer after a batched prefill.
    ///
    /// `raw` holds the last `qkv_conv_taps - 1` **pre-convolution** vectors for
    /// this projection, oldest first. The batched path convolves in place, so
    /// the caller has to keep these before it does.
    pub fn seed_conv_tail(&mut self, li: usize, which: Qkv, raw: &[f32]) {
        let layer = &mut self.layers[li];
        let tail = match which {
            Qkv::Q => &mut layer.q_tail,
            Qkv::K => &mut layer.k_tail,
            Qkv::V => &mut layer.v_tail,
        };
        debug_assert_eq!(raw.len(), tail.len());
        tail.copy_from_slice(raw);
    }

    /// Seed one site's Engram value tail, oldest first, pre-convolution.
    pub fn seed_engram_tail(&mut self, site: usize, raw: &[f32]) {
        let tail = &mut self.engram_tail[site];
        debug_assert_eq!(raw.len(), tail.len());
        tail.copy_from_slice(raw);
    }

    /// Seed the token history the n-gram hash reads, oldest first.
    pub fn seed_tokens(&mut self, tokens: &[u32]) {
        let keep = self.cfg.engram.orders.iter().copied().max().unwrap_or(1);
        self.tokens.clear();
        let start = tokens.len().saturating_sub(keep);
        self.tokens.extend_from_slice(&tokens[start..]);
    }

    /// Set the next write position after a batched prefill.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Advance to the next position. Call once per decoded token, after every
    /// layer has written its key and value.
    pub fn advance(&mut self) {
        self.pos += 1;
    }

    /// Current write position.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// Which projection a conv step is rolling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qkv {
    Q,
    K,
    V,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v3::config::V3Engram;

    fn cfg() -> V3Config {
        V3Config {
            vocab_size: 8192,
            out_vocab: 8192,
            d_model: 8,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 4,
            qk_head_dim: 4,
            v_head_dim: 4,
            max_seq_len: 64,
            hada_n: 8,
            mhc_lanes: 2,
            rope_theta: 10000.0,
            sliding_window: 4,
            global_layers: vec![3],
            qkv_conv_taps: 3,
            engram: V3Engram {
                orders: vec![2, 3],
                heads: 1,
                slots: 16,
                sub_dim: 4,
                sites: vec![1],
                conv_taps: 4,
                conv_dilation: 3,
                seed_heads: 0,
            },
        }
    }

    #[test]
    fn layers_grow_to_their_cap_and_no_further() {
        let c = cfg();
        let mut cache = V3Cache::new(&c, 2);
        assert_eq!(
            cache.slots(0),
            2,
            "local layer starts at the hint, not its window"
        );
        assert_eq!(cache.slots(3), 2, "global layer starts at the hint");

        let k = vec![1.0f32; c.k_dim()];
        let v = vec![1.0f32; c.v_dim()];

        // Layer 0 is local with a window of 4: it grows to 4 and then wraps.
        cache.write_kv(0, 3, &k, &v);
        assert_eq!(cache.slots(0), 4);
        cache.write_kv(0, 99, &k, &v);
        assert_eq!(
            cache.slots(0),
            4,
            "a local layer must never exceed its window"
        );

        // Layer 3 is global: it keeps growing.
        cache.write_kv(3, 9, &k, &v);
        assert!(cache.slots(3) >= 10, "global layer should have grown");
    }

    /// The geometry the shipped needle3.cact declares, so the byte counts
    /// below are the ones a real session pays.
    fn shipped() -> V3Config {
        V3Config {
            vocab_size: 8192,
            out_vocab: 8192,
            d_model: 768,
            num_heads: 12,
            num_kv_heads: 2,
            num_layers: 20,
            qk_head_dim: 48,
            v_head_dim: 64,
            max_seq_len: 8192,
            hada_n: 1024,
            mhc_lanes: 4,
            rope_theta: 100_000.0,
            sliding_window: 1024,
            global_layers: vec![4, 9, 14, 19],
            qkv_conv_taps: 3,
            engram: V3Engram {
                orders: vec![2, 3],
                heads: 3,
                slots: 18432,
                sub_dim: 128,
                sites: vec![3, 7, 11, 15, 19],
                conv_taps: 4,
                conv_dilation: 3,
                seed_heads: 0,
            },
        }
    }

    #[test]
    fn a_global_layer_never_asks_for_more_than_it_stores() {
        // Past max_seq_len the ring wraps, and an unclamped span would make a
        // position read itself as its own past — silently in release, where
        // the bounds check is compiled out.
        let mut c = shipped();
        c.max_seq_len = 8;
        c.sliding_window = 4;
        c.num_layers = 2;
        c.global_layers = vec![1];
        let mut cache = V3Cache::new(&c, 8);
        let k = vec![0.0f32; c.k_dim()];
        let v = vec![0.0f32; c.v_dim()];

        for pos in 0..12 {
            cache.write_kv(1, pos, &k, &v);
            let (lo, hi) = cache.span(1, pos);
            let slots = cache.slots(1);
            assert!(
                hi - lo < slots,
                "span {lo}..={hi} exceeds {slots} slots at position {pos}"
            );
        }
        // And the caller can see that history was lost.
        for _ in 0..12 {
            cache.advance();
        }
        assert!(
            cache.is_saturated(),
            "running past the context must be visible"
        );
    }

    #[test]
    fn a_short_session_pays_for_what_it_uses() {
        let c = shipped();
        let k = vec![0.0f32; c.k_dim()];
        let v = vec![0.0f32; c.v_dim()];

        let mut short = V3Cache::new(&c, 512);
        for li in 0..c.num_layers {
            short.write_kv(li, 511, &k, &v);
        }
        let short_mb = short.bytes() as f64 / 1024.0 / 1024.0;

        // Reserving every window and the full context instead, which is what
        // a fixed allocation would cost.
        let reserved = (16 * 1024 + 4 * 8192) * (c.k_dim() + c.v_dim()) * 4;
        let reserved_mb = reserved as f64 / 1024.0 / 1024.0;

        assert!(
            short_mb < reserved_mb / 4.0,
            "a 512-token session should cost far less than reserving the \
             windows: {short_mb:.1} MB vs {reserved_mb:.1} MB"
        );
        // 8.75 MB at f32: 20 layers x 512 slots x (96 + 128) x 4 bytes.
        //
        // The container declares kv_bits = 8, so upstream post-trained this
        // model for an int8 KV cache, which would be 2.2 MB. Storing f32 is
        // deliberate for now: it is what keeps decode bit-identical to the
        // verified prefill, and the reference's own decode path runs
        // unquantised. Quantising the cache is a real 4x saving on constrained
        // targets but needs its own parity story against the reference's
        // quant=True path, so it is a separate piece of work rather than a
        // silent numerics change.
        assert!(
            short_mb < 9.5,
            "expected under 9.5 MB at f32, got {short_mb:.1} MB"
        );
        assert!(
            short_mb > 8.0,
            "unexpectedly small — has the KV representation changed? {short_mb:.1} MB"
        );
    }

    #[test]
    fn a_long_session_caps_the_local_layers() {
        let c = shipped();
        let k = vec![0.0f32; c.k_dim()];
        let v = vec![0.0f32; c.v_dim()];
        let mut cache = V3Cache::new(&c, 64);
        for li in 0..c.num_layers {
            cache.write_kv(li, 4095, &k, &v);
        }
        for li in 0..c.num_layers {
            if c.is_global(li) {
                assert!(
                    cache.slots(li) >= 4096,
                    "global layer {li} must hold all of it"
                );
            } else {
                assert_eq!(cache.slots(li), 1024, "local layer {li} caps at its window");
            }
        }
    }

    #[test]
    fn spans_bound_local_layers_only() {
        let c = cfg();
        let mut cache = V3Cache::new(&c, 8);
        let k = vec![0.0f32; c.k_dim()];
        let v = vec![0.0f32; c.v_dim()];
        // Written before queried, as the decode path does — a global layer's
        // ring grows on write, and the span is clamped to what it holds.
        for pos in 0..=10 {
            cache.write_kv(0, pos, &k, &v);
            cache.write_kv(3, pos, &k, &v);
        }
        // Layer 0 is local with a window of 4.
        assert_eq!(cache.span(0, 2), (0, 2));
        assert_eq!(cache.span(0, 10), (7, 10));
        // Layer 3 is global: it kept everything, so it sees everything.
        assert_eq!(cache.span(3, 10), (0, 10));
    }

    #[test]
    fn conv_step_matches_the_sequence_conv() {
        use crate::v3::attention::causal_depthwise_conv;
        let c = cfg();
        let dim = c.q_dim();
        let taps: Vec<f32> = (0..3 * dim).map(|i| (i as f32 * 0.37).sin()).collect();

        // Reference: the whole sequence at once, already parity-verified.
        let seq = 6;
        let raw: Vec<f32> = (0..seq * dim).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut want = raw.clone();
        causal_depthwise_conv(&mut want, &taps, seq, dim, 3);

        // Step form, one position at a time.
        let mut cache = V3Cache::new(&c, seq);
        for t in 0..seq {
            let mut cur = raw[t * dim..(t + 1) * dim].to_vec();
            cache.conv_step(0, Qkv::Q, &taps, &mut cur);
            for (i, &g) in cur.iter().enumerate() {
                let e = want[t * dim + i];
                assert!(
                    (g - e).abs() < 1e-5,
                    "position {t} dim {i}: step {g} vs sequence {e}"
                );
            }
            cache.advance();
        }
    }

    #[test]
    fn engram_conv_step_matches_the_sequence_conv() {
        use crate::v3::engram::{value_conv, EngramDims};
        let c = cfg();
        let d = c.d_model;
        let dims = EngramDims {
            num_tables: 2,
            slots: 16,
            sub_dim: 4,
            d_model: d,
            conv_taps: 4,
            conv_dilation: 3,
            max_order: 3,
        };
        let taps: Vec<f32> = (0..4 * d).map(|i| (i as f32 * 0.23).cos()).collect();
        let seq = 12;
        let raw: Vec<f32> = (0..seq * d).map(|i| (i as f32 * 0.19).sin()).collect();
        let mut want = raw.clone();
        value_conv(&mut want, &taps, seq, &dims);

        let mut cache = V3Cache::new(&c, seq);
        for t in 0..seq {
            let mut cur = raw[t * d..(t + 1) * d].to_vec();
            cache.engram_conv_step(0, &taps, &mut cur);
            for (i, &g) in cur.iter().enumerate() {
                let e = want[t * d + i];
                assert!(
                    (g - e).abs() < 1e-5,
                    "position {t} dim {i}: step {g} vs sequence {e}"
                );
            }
            cache.advance();
        }
    }
}
