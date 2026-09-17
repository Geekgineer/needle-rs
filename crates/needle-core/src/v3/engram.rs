//! Needle 3 Engram — n-gram hash memory.
//!
//! Five sites in v3 against v2's two, six tables against four, and 18432 slots
//! against 8192. The structure is the same shape as v2's but none of the
//! constants are, so it is written fresh rather than parameterised over v2.
//!
//! Each table hashes a different n-gram order and hash head. A position's
//! n-gram is only valid once enough tokens exist, and the fetched rows are
//! masked to zero before that — otherwise the first positions would index on
//! padding and pull real rows out of the table.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::math::sqrt;
use crate::ops::sigmoid;

/// FNV-style mixing constants, from upstream `architecture.py`.
const ENGRAM_SEED: u32 = 0x9E37_79B9;
const ENGRAM_PRIME: u32 = 0x0100_0193;

/// Hash every position into one slot per table.
///
/// Returns `(seq, num_tables)` row-major, with table `oi * heads + h` covering
/// n-gram order `orders[oi]` under hash head `h`.
///
/// Positions before an n-gram is complete still hash — they mix in token id 0
/// for the missing history, exactly as upstream's zero-padded shift does — and
/// are masked out afterwards by [`ngram_valid`].
pub fn engram_indices(
    tokens: &[u32],
    orders: &[usize],
    heads: usize,
    slots: u32,
    seed_heads: usize,
) -> Vec<u32> {
    let stride = if seed_heads == 0 { heads } else { seed_heads };
    let seq = tokens.len();
    let num_tables = orders.len() * heads;
    let mut out = vec![0u32; seq * num_tables];

    for (oi, &order) in orders.iter().enumerate() {
        for h in 0..heads {
            let table = oi * heads + h;
            let seed = ENGRAM_SEED.wrapping_mul((oi * stride + h + 1) as u32);
            for p in 0..seq {
                let mut acc = seed;
                for j in 0..order {
                    // Upstream shifts right with zero padding, so history
                    // before the start reads as token 0.
                    let tok = if j > p { 0 } else { tokens[p - j] };
                    acc = (acc ^ tok).wrapping_mul(ENGRAM_PRIME);
                }
                acc ^= acc >> 15;
                out[p * num_tables + table] = acc % slots;
            }
        }
    }
    out
}

/// `true` where position `p` has enough history for table `t`'s n-gram order.
///
/// Order `o` needs `o` tokens, so it is valid from position `o - 1`.
pub fn ngram_valid(p: usize, table: usize, orders: &[usize], heads: usize) -> bool {
    let order = orders[table / heads];
    p + 1 >= order
}

/// Weights for one Engram site. Projections stay as the caller's
/// representation; only the table rows and taps are dense here.
pub struct EngramDims {
    pub num_tables: usize,
    pub slots: usize,
    pub sub_dim: usize,
    pub d_model: usize,
    pub conv_taps: usize,
    pub conv_dilation: usize,
    /// Largest n-gram order — the stride upstream uses for the tap mask.
    pub max_order: usize,
}

/// The causal dilated convolution applied to the Engram value stream.
///
/// `v_out[p] = Σ_j taps[j] · v[p - j·dilation]`, zero before the start. The
/// tap is additionally gated off until `p >= j · max_order`, which is how
/// upstream masks it.
pub fn value_conv(buf: &mut [f32], taps: &[f32], seq: usize, d: &EngramDims) {
    let dim = d.d_model;
    debug_assert_eq!(buf.len(), seq * dim);
    debug_assert_eq!(taps.len(), d.conv_taps * dim);
    for p in (0..seq).rev() {
        for c in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..d.conv_taps {
                let back = j * d.conv_dilation;
                if back > p || p < j * d.max_order {
                    continue;
                }
                acc += taps[j * dim + c] * buf[(p - back) * dim + c];
            }
            buf[p * dim + c] = acc;
        }
    }
}

/// The gate that folds one site's memory into the residual stream.
///
/// `alpha = sigmoid(⟨rms_unit(x), rms_unit(k)⟩ / sqrt(d_model))`, then
/// `x += alpha · v`. `rms_unit` normalises without a learned scale.
pub fn apply_site(x: &mut [f32], k: &[f32], v: &[f32], d_model: usize) {
    debug_assert_eq!(x.len(), d_model);
    let unit = |z: &[f32]| -> f32 {
        let mut sq = 0.0f32;
        for &a in z {
            sq += a * a;
        }
        1.0 / sqrt(sq / d_model as f32 + 1e-6)
    };
    let ux = unit(x);
    let uk = unit(k);

    let mut dot = 0.0f32;
    for (a, b) in x.iter().zip(k) {
        dot += a * b;
    }
    let alpha = sigmoid(dot * ux * uk / sqrt(d_model as f32));
    for (xi, &vi) in x.iter_mut().zip(v) {
        *xi += alpha * vi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_bigram_depends_on_exactly_two_tokens() {
        let orders = vec![2usize];
        // Same current token, different predecessor, so a correct bigram hash
        // must differ; a unigram hash would not.
        let a = engram_indices(&[10, 20, 30], &orders, 1, 18432, 0);
        let b = engram_indices(&[10, 99, 30], &orders, 1, 18432, 0);
        assert_eq!(a[0], b[0], "position 0 sees only token 10 either way");
        assert_ne!(a[1], b[1], "position 1's bigram changed");
        assert_ne!(a[2], b[2], "position 2's bigram changed");
    }

    #[test]
    fn tables_differ_by_seed() {
        let orders = vec![2usize, 3];
        let idx = engram_indices(&[7, 8, 9], &orders, 3, 18432, 0);
        let row: Vec<u32> = (0..6).map(|t| idx[2 * 6 + t]).collect();
        // Six tables, six seeds: collisions are possible but all six equal
        // would mean the seed is not reaching the hash.
        assert!(
            row.windows(2).any(|w| w[0] != w[1]),
            "all six tables produced the same slot: {row:?}"
        );
    }

    #[test]
    fn validity_follows_the_order() {
        let orders = vec![2usize, 3];
        // heads = 1, so table 0 is order 2 and table 1 is order 3.
        assert!(!ngram_valid(0, 0, &orders, 1));
        assert!(ngram_valid(1, 0, &orders, 1));
        assert!(!ngram_valid(1, 1, &orders, 1));
        assert!(ngram_valid(2, 1, &orders, 1));
    }
}
