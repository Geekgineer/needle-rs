//! Needle 3 geometry.
//!
//! Everything here is read from the container header, never assumed. The
//! differences that matter against v2 are asymmetric query/value head widths,
//! a per-layer choice between sliding and global attention, and a causal
//! depthwise convolution over Q, K and V.

extern crate alloc;
use alloc::vec::Vec;

/// Engram placement and table layout.
#[derive(Debug, Clone, PartialEq)]
pub struct V3Engram {
    /// N-gram orders, `(2, 3)` on the shipped model.
    pub orders: Vec<usize>,
    /// Hash heads per order. `num_tables == orders.len() * heads`.
    pub heads: usize,
    /// Rows per table.
    pub slots: usize,
    /// Width of one table row, `d_model / (orders.len() * heads)`.
    pub sub_dim: usize,
    /// Layers carrying an Engram, `(3, 7, 11, 15, 19)` on the shipped model.
    pub sites: Vec<usize>,
    pub conv_taps: usize,
    pub conv_dilation: usize,
    pub seed_heads: usize,
}

impl V3Engram {
    /// Total hash tables: one per (order, head) pair.
    pub fn num_tables(&self) -> usize {
        self.orders.len() * self.heads
    }

    /// Index of `layer` among the Engram sites, if it carries one.
    pub fn site_of(&self, layer: usize) -> Option<usize> {
        self.sites.iter().position(|&l| l == layer)
    }
}

/// Full v3 model geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct V3Config {
    pub vocab_size: usize,
    /// Rows of the tied head actually used for logits. Rows beyond this are
    /// input-only code embeddings. `0` means the full vocabulary.
    pub out_vocab: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    /// Query and key head width — 48 on the shipped model.
    pub qk_head_dim: usize,
    /// Value head width — 64. Unlike v2, this differs from `qk_head_dim`.
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    /// Hadamard width in HadamardMLP: `d_model` rounded up to a power of two.
    pub hada_n: usize,
    pub mhc_lanes: usize,
    pub rope_theta: f32,
    /// Local attention width for layers that are not global. `0` means full
    /// causal everywhere.
    pub sliding_window: usize,
    /// Layers attending over the whole sequence instead of `sliding_window`.
    pub global_layers: Vec<usize>,
    /// Causal depthwise convolution width over Q, K and V. `0` means none.
    pub qkv_conv_taps: usize,
    pub engram: V3Engram,
}

impl V3Config {
    /// Width of the concatenated query projection, `num_heads * qk_head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.qk_head_dim
    }

    /// Width of the concatenated key projection.
    pub fn k_dim(&self) -> usize {
        self.num_kv_heads * self.qk_head_dim
    }

    /// Width of the concatenated value projection.
    pub fn v_dim(&self) -> usize {
        self.num_kv_heads * self.v_head_dim
    }

    /// Width of the attention output before `out_proj`, `num_heads * v_head_dim`.
    pub fn attn_out_dim(&self) -> usize {
        self.num_heads * self.v_head_dim
    }

    /// Query heads served by one key/value head.
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Flattened mHC lane-stream width, `mhc_lanes * d_model`.
    pub fn mhc_width(&self) -> usize {
        self.mhc_lanes * self.d_model
    }

    /// Rows of the tied head used for logits.
    pub fn logit_rows(&self) -> usize {
        if self.out_vocab == 0 {
            self.vocab_size
        } else {
            self.out_vocab
        }
    }

    /// True when `layer` attends over the whole sequence.
    pub fn is_global(&self, layer: usize) -> bool {
        self.global_layers.contains(&layer)
    }

    /// How many past positions `layer` can attend to, counting the current
    /// one. `None` means unbounded — the caller must keep full history.
    ///
    /// This is the distinction that drives the KV allocation: local layers
    /// need a fixed ring, global layers need everything.
    pub fn attention_span(&self, layer: usize) -> Option<usize> {
        if self.is_global(layer) || self.sliding_window == 0 {
            None
        } else {
            Some(self.sliding_window)
        }
    }

    /// The lane layer `i` is assigned to, as `eye(n)[i % n]` in the
    /// reference `Stack`.
    pub fn active_lane(&self, layer: usize) -> usize {
        layer % self.mhc_lanes
    }

    /// Bytes of key/value cache for a session of `seq_len` positions, at
    /// `bytes_per` per stored scalar.
    ///
    /// Local layers cost a fixed ring; global layers cost the real sequence
    /// length, which is why they are sized to what a session actually uses
    /// rather than to `max_seq_len`.
    pub fn kv_bytes(&self, seq_len: usize, bytes_per: usize) -> usize {
        let per_pos = self.num_kv_heads * (self.qk_head_dim + self.v_head_dim);
        let mut total = 0;
        for layer in 0..self.num_layers {
            let slots = match self.attention_span(layer) {
                Some(w) => seq_len.min(w),
                None => seq_len,
            };
            total += slots * per_pos * bytes_per;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The geometry the shipped `needle3.cact` declares.
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
    fn projection_widths_match_the_container() {
        let c = shipped();
        // Verified against the shipped directory: q_proj is 576x768,
        // k_proj 96x768, v_proj 128x768.
        assert_eq!(c.q_dim(), 576);
        assert_eq!(c.k_dim(), 96);
        assert_eq!(c.v_dim(), 128);
        assert_eq!(c.attn_out_dim(), 768);
        assert_eq!(c.kv_repeat(), 6);
    }

    #[test]
    fn engram_tables_are_orders_times_heads() {
        // The container reports num_engram_tables 6 while upstream's
        // engram_geometry returns 3; the 3 is the head count. Upstream:
        // heads = d_model // (len(orders) * 128) = 768 // 256 = 3.
        let c = shipped();
        assert_eq!(c.engram.num_tables(), 6);
        assert_eq!(c.d_model / (c.engram.orders.len() * c.engram.sub_dim), 3);
        assert_eq!(c.engram.site_of(11), Some(2));
        assert_eq!(c.engram.site_of(12), None);
    }

    #[test]
    fn global_layers_have_no_attention_span() {
        let c = shipped();
        for l in [4, 9, 14, 19] {
            assert!(c.is_global(l), "layer {l} should be global");
            assert_eq!(c.attention_span(l), None, "layer {l} span");
        }
        for l in [0, 3, 10, 18] {
            assert!(!c.is_global(l));
            assert_eq!(c.attention_span(l), Some(1024), "layer {l} span");
        }
    }

    #[test]
    fn kv_cost_is_paid_for_what_a_session_uses() {
        let c = shipped();
        // A typical tool-calling session is a few hundred positions. Sizing
        // the global layers to the sequence rather than to max_seq_len is the
        // difference between a browser-viable session and a 28 MB cache.
        let short = c.kv_bytes(512, 1);
        let full = c.kv_bytes(8192, 1);
        // Measured: 2.2 MB against 10.5 MB at int8. Sizing to the session
        // rather than to max_seq_len is a 4.8x saving on a typical prompt.
        assert_eq!(short, 512 * 20 * 224);
        assert_eq!(full, (16 * 1024 + 4 * 8192) * 224);
        assert!(
            short * 4 < full,
            "a 512-token session should cost far less than a full-context one: \
             {short} vs {full}"
        );
        // Local layers saturate at the window; only the 4 global ones grow.
        let at_window = c.kv_bytes(1024, 1);
        let beyond = c.kv_bytes(2048, 1);
        let per_pos = c.num_kv_heads * (c.qk_head_dim + c.v_head_dim);
        assert_eq!(
            beyond - at_window,
            c.global_layers.len() * 1024 * per_pos,
            "past the window only the global layers should keep growing"
        );
    }
}
