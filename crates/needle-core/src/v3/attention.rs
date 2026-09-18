//! Needle 3 attention kernels.
//!
//! Three things differ from v2 and each is isolated here so it can be
//! verified on its own:
//!
//! * **Asymmetric head widths.** Queries and keys are 48 wide, values 64.
//!   The reference pads both to `max(qk, v)` to reach a fused flash kernel
//!   and pre-scales the query by `sqrt(fused/qk)` to cancel the wider
//!   `1/sqrt(fused)` that kernel applies. That is a padding trick, not
//!   mathematics: the effective scale is `1/sqrt(qk_head_dim)`, which is
//!   what this computes directly.
//! * **A causal depthwise convolution** over Q, K and V along the sequence.
//! * **Per-layer attention span.** Sixteen layers see a 1024-position window,
//!   four see everything.
//!
//! Weights arrive already projected, so the caller does the matvecs with
//! whatever representation it holds — packed `CqWeight` in production, dense
//! floats in the component tests — and these kernels stay pure.

extern crate alloc;
use alloc::vec;

use crate::math::{exp, sqrt};

/// How a cache hands its keys and values to attention.
///
/// The int8 form stores one `i8` per element plus an `f32` scale per head
/// vector. Because the scale is constant across a head, it hoists out of the
/// dot product entirely — `Σ (q·scale)·x` is `scale · Σ q·x` — so reading
/// quantised costs an integer-to-float conversion per element and nothing else.
#[derive(Debug, Clone, Copy)]
pub enum KvStore<'a> {
    F32 {
        k: &'a [f32],
        v: &'a [f32],
    },
    Int8 {
        k: &'a [i8],
        /// One scale per `(slot, kv_head)`.
        k_scale: &'a [f32],
        v: &'a [i8],
        v_scale: &'a [f32],
    },
}

/// A ring-buffered key/value cache and the window into it a query may read.
///
/// `k` and `v` are `(slots, num_kv_heads, ·)` and logical position `p` lives
/// at slot `p % slots`. `lo..=hi` is inclusive.
#[derive(Debug, Clone, Copy)]
pub struct Ring<'a> {
    pub kv: KvStore<'a>,
    pub slots: usize,
    pub lo: usize,
    pub hi: usize,
}

/// Shapes one attention call works over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnDims {
    pub seq: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
}

impl AttnDims {
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Causal depthwise convolution along the sequence, in place.
///
/// `out[t][c] = Σ_j taps[j][c] · z[t - j][c]`, with positions before the start
/// contributing zero. `buf` is `(seq, dim)` row-major and `taps` is
/// `(n_taps, dim)`.
///
/// Walks `t` downwards so each position reads only earlier ones, which have
/// not been rewritten yet — no scratch copy needed.
pub fn causal_depthwise_conv(buf: &mut [f32], taps: &[f32], seq: usize, dim: usize, n_taps: usize) {
    debug_assert_eq!(buf.len(), seq * dim);
    debug_assert_eq!(taps.len(), n_taps * dim);
    if n_taps == 0 {
        return;
    }
    for t in (0..seq).rev() {
        for c in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..n_taps {
                if j > t {
                    break;
                }
                acc += taps[j * dim + c] * buf[(t - j) * dim + c];
            }
            buf[t * dim + c] = acc;
        }
    }
}

/// `ZCRMSNorm` then RoPE, over a `(seq, heads, head_dim)` buffer, in place.
///
/// The norm is `(1 + scale) · x / sqrt(mean(x²) + 1e-6)` — despite the name it
/// subtracts no mean. RoPE uses the half-split convention: the first half of
/// each head rotates against the second, not adjacent pairs.
pub fn norm_and_rope(
    buf: &mut [f32],
    scale: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(buf.len(), seq * heads * head_dim);
    debug_assert_eq!(scale.len(), head_dim);
    let half = head_dim / 2;
    for t in 0..seq {
        for h in 0..heads {
            let off = (t * heads + h) * head_dim;
            let x = &mut buf[off..off + head_dim];

            let mut sq = 0.0f32;
            for &v in x.iter() {
                sq += v * v;
            }
            let rms = sqrt(sq / head_dim as f32 + 1e-6);
            for (v, &s) in x.iter_mut().zip(scale) {
                *v = (1.0 + s) * *v / rms;
            }

            for i in 0..half {
                let c = cos[t * half + i];
                let s = sin[t * half + i];
                let x1 = x[i];
                let x2 = x[half + i];
                x[i] = x1 * c - x2 * s;
                x[half + i] = x2 * c + x1 * s;
            }
        }
    }
}

/// Grouped-query attention over a whole sequence.
///
/// `q` is `(seq, num_heads, qk)`, `k` is `(seq, num_kv_heads, qk)`, `v` is
/// `(seq, num_kv_heads, v)`, all row-major. `out` is
/// `(seq, num_heads * v_head_dim)`.
///
/// `window` bounds how far back a query may look, counting itself; `None`
/// attends over the whole causal prefix. That is the only difference between
/// v3's local and global layers.
pub fn attend(q: &[f32], kv: KvStore<'_>, d: AttnDims, window: Option<usize>, out: &mut [f32]) {
    debug_assert_eq!(q.len(), d.seq * d.num_heads * d.qk_head_dim);
    debug_assert_eq!(out.len(), d.seq * d.num_heads * d.v_head_dim);

    let scale = 1.0f32 / sqrt(d.qk_head_dim as f32);
    let repeat = d.kv_repeat();
    let mut scores = vec![0.0f32; d.seq];

    for t in 0..d.seq {
        // Inclusive lower bound on which positions this query may attend to.
        let lo = match window {
            Some(w) if t + 1 > w => t + 1 - w,
            _ => 0,
        };
        for h in 0..d.num_heads {
            let kvh = h / repeat;
            let qo = (t * d.num_heads + h) * d.qk_head_dim;
            let qv = &q[qo..qo + d.qk_head_dim];

            let mut max = f32::NEG_INFINITY;
            for (n, s) in scores.iter_mut().enumerate().take(t + 1).skip(lo) {
                let ko = (n * d.num_kv_heads + kvh) * d.qk_head_dim;
                let acc = match kv {
                    KvStore::F32 { k, .. } => {
                        let mut a = 0.0f32;
                        for (x, y) in qv.iter().zip(&k[ko..ko + d.qk_head_dim]) {
                            a += x * y;
                        }
                        a
                    }
                    KvStore::Int8 { k, k_scale, .. } => {
                        let mut a = 0.0f32;
                        for (x, y) in qv.iter().zip(&k[ko..ko + d.qk_head_dim]) {
                            a += x * (*y as f32);
                        }
                        a * k_scale[n * d.num_kv_heads + kvh]
                    }
                };
                *s = acc * scale;
                if *s > max {
                    max = *s;
                }
            }

            let mut sum = 0.0f32;
            for s in scores.iter_mut().take(t + 1).skip(lo) {
                *s = exp(*s - max);
                sum += *s;
            }
            let inv = 1.0 / sum;

            let oo = (t * d.num_heads + h) * d.v_head_dim;
            let o = &mut out[oo..oo + d.v_head_dim];
            o.fill(0.0);
            for (n, &s) in scores.iter().enumerate().take(t + 1).skip(lo) {
                let w = s * inv;
                let vo = (n * d.num_kv_heads + kvh) * d.v_head_dim;
                match kv {
                    KvStore::F32 { v, .. } => {
                        for (oi, &vi) in o.iter_mut().zip(&v[vo..vo + d.v_head_dim]) {
                            *oi += w * vi;
                        }
                    }
                    KvStore::Int8 { v, v_scale, .. } => {
                        let w = w * v_scale[n * d.num_kv_heads + kvh];
                        for (oi, &vi) in o.iter_mut().zip(&v[vo..vo + d.v_head_dim]) {
                            *oi += w * (vi as f32);
                        }
                    }
                }
            }
        }
    }
}

/// One decode step against a ring-buffered cache.
///
/// `q` is this position's queries, `(num_heads, qk_head_dim)`. `k` and `v` are
/// the cache, `(slots, num_kv_heads, ·)`, where logical position `p` lives at
/// slot `p % slots`. `lo..=hi` is the inclusive range of logical positions this
/// query may attend to — the caller derives it from the layer's span, which is
/// the whole of the local/global distinction.
///
/// A local layer passes `slots = sliding_window` and a bounded `lo`; a global
/// layer passes `slots >= hi + 1` and `lo = 0`. Nothing here needs to know
/// which kind it is serving.
pub fn attend_step(q: &[f32], ring: Ring<'_>, d: AttnDims, out: &mut [f32]) {
    let Ring { kv, slots, lo, hi } = ring;
    debug_assert_eq!(q.len(), d.num_heads * d.qk_head_dim);
    debug_assert_eq!(out.len(), d.num_heads * d.v_head_dim);
    debug_assert!(lo <= hi, "empty attention range {lo}..={hi}");
    debug_assert!(hi - lo < slots, "range {lo}..={hi} exceeds {slots} slots");

    let scale = 1.0f32 / sqrt(d.qk_head_dim as f32);
    let repeat = d.kv_repeat();
    let span = hi - lo + 1;
    let mut scores = vec![0.0f32; span];

    for h in 0..d.num_heads {
        let kvh = h / repeat;
        let qv = &q[h * d.qk_head_dim..(h + 1) * d.qk_head_dim];

        let mut max = f32::NEG_INFINITY;
        for (i, s) in scores.iter_mut().enumerate() {
            let slot = (lo + i) % slots;
            let ko = (slot * d.num_kv_heads + kvh) * d.qk_head_dim;
            let acc = match kv {
                KvStore::F32 { k, .. } => {
                    let mut a = 0.0f32;
                    for (x, y) in qv.iter().zip(&k[ko..ko + d.qk_head_dim]) {
                        a += x * y;
                    }
                    a
                }
                KvStore::Int8 { k, k_scale, .. } => {
                    // The scale is constant across a head vector, so it comes
                    // out of the inner loop: sum(q * i8) * scale.
                    let mut a = 0.0f32;
                    for (x, y) in qv.iter().zip(&k[ko..ko + d.qk_head_dim]) {
                        a += x * (*y as f32);
                    }
                    a * k_scale[slot * d.num_kv_heads + kvh]
                }
            };
            *s = acc * scale;
            if *s > max {
                max = *s;
            }
        }

        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = exp(*s - max);
            sum += *s;
        }
        let inv = 1.0 / sum;

        let o = &mut out[h * d.v_head_dim..(h + 1) * d.v_head_dim];
        o.fill(0.0);
        for (i, &s) in scores.iter().enumerate() {
            let w = s * inv;
            let slot = (lo + i) % slots;
            let vo = (slot * d.num_kv_heads + kvh) * d.v_head_dim;
            match kv {
                KvStore::F32 { v, .. } => {
                    for (oi, &vi) in o.iter_mut().zip(&v[vo..vo + d.v_head_dim]) {
                        *oi += w * vi;
                    }
                }
                KvStore::Int8 { v, v_scale, .. } => {
                    let w = w * v_scale[slot * d.num_kv_heads + kvh];
                    for (oi, &vi) in o.iter_mut().zip(&v[vo..vo + d.v_head_dim]) {
                        *oi += w * (vi as f32);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_is_causal_and_zero_padded() {
        // dim 1, two taps: out[t] = a*z[t] + b*z[t-1], out[0] = a*z[0].
        let mut buf = [1.0f32, 2.0, 3.0];
        let taps = [10.0f32, 100.0];
        causal_depthwise_conv(&mut buf, &taps, 3, 1, 2);
        assert_eq!(buf, [10.0, 20.0 + 100.0, 30.0 + 200.0]);
    }

    #[test]
    fn conv_with_zero_taps_is_identity() {
        let mut buf = [1.0f32, 2.0];
        causal_depthwise_conv(&mut buf, &[], 2, 1, 0);
        assert_eq!(buf, [1.0, 2.0]);
    }

    #[test]
    fn attend_step_wraps_the_ring() {
        // Three slots, and we sit at logical position 4 with a window of 3,
        // so the attendable range is 2..=4 and lands on slots 2, 0, 1.
        let d = AttnDims {
            seq: 1,
            num_heads: 1,
            num_kv_heads: 1,
            qk_head_dim: 1,
            v_head_dim: 1,
        };
        let q = [0.0f32]; // zero query, so every score ties and weights are 1/3
                          // slot 0 holds position 3, slot 1 position 4, slot 2 position 2.
        let k = [1.0f32, 1.0, 1.0];
        let v = [30.0f32, 40.0, 20.0];
        let mut out = [0.0f32];
        attend_step(
            &q,
            Ring {
                kv: KvStore::F32 { k: &k, v: &v },
                slots: 3,
                lo: 2,
                hi: 4,
            },
            d,
            &mut out,
        );
        assert!(
            (out[0] - 30.0).abs() < 1e-5,
            "mean of 20, 30, 40 expected, got {}",
            out[0]
        );
    }

    #[test]
    fn a_window_of_one_attends_only_to_itself() {
        // Two positions, one head, head_dim 1. With window 1 the second
        // position must ignore the first entirely.
        let d = AttnDims {
            seq: 2,
            num_heads: 1,
            num_kv_heads: 1,
            qk_head_dim: 1,
            v_head_dim: 1,
        };
        let q = [1.0f32, 1.0];
        let k = [1.0f32, 1.0];
        let v = [5.0f32, 9.0];
        let mut out = [0.0f32; 2];
        attend(&q, KvStore::F32 { k: &k, v: &v }, d, Some(1), &mut out);
        assert_eq!(out, [5.0, 9.0]);

        // Unbounded, both positions are equally weighted at position 1.
        let mut out2 = [0.0f32; 2];
        attend(&q, KvStore::F32 { k: &k, v: &v }, d, None, &mut out2);
        assert_eq!(out2[0], 5.0);
        assert!((out2[1] - 7.0).abs() < 1e-6, "got {}", out2[1]);
    }
}
