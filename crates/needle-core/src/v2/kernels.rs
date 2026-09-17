//! Kernels specific to Needle v2: `_rms_unit`, Sinkhorn, HadamardMLP, and the
//! Engram hash.
//!
//! Each is a port of a named function in the upstream reference; the doc comment
//! says which.

use crate::hadamard::fwht_normalized;

/// SiLU / swish. Lives in [`crate::ops`] now that v3 uses it too; re-exported
/// here so existing call sites are unchanged.
pub use crate::ops::silu;

/// `decode._hadamard` — the HadamardMLP body.
///
/// ```text
/// z = (d1 * z) @ H
/// z = silu(d2 * z) @ H
/// out = (d3 * z)[:d_model]
/// ```
///
/// `H` is `Walsh(hada_n) / sqrt(hada_n)`, which is symmetric — so `z @ H` equals
/// `H @ z` and both are one normalised FWHT. That turns each of the two dense
/// `hada_n x hada_n` matmuls into `hada_n * log2(hada_n)` add/sub.
///
/// `scratch` must be `hada_n` long; `x` and `out` are `d_model` long. `x` is
/// zero-extended into the scratch when `hada_n > d_model`, which is the padding
/// upstream applies.
pub fn hadamard_mlp(
    x: &[f32],
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    scratch: &mut [f32],
    out: &mut [f32],
) {
    let n = scratch.len();
    let d = x.len();
    debug_assert!(n >= d && n.is_power_of_two());
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    debug_assert_eq!(out.len(), d);

    for i in 0..n {
        scratch[i] = if i < d { x[i] * d1[i] } else { 0.0 };
    }
    fwht_normalized(scratch);
    for i in 0..n {
        scratch[i] = silu(scratch[i] * d2[i]);
    }
    fwht_normalized(scratch);
    for i in 0..d {
        out[i] = scratch[i] * d3[i];
    }
}

/// Seed constant in `architecture.engram_indices`.
pub const ENGRAM_SEED: u32 = 0x9E37_79B9;
/// FNV-style multiplier in `architecture.engram_indices`.
pub const ENGRAM_PRIME: u32 = 0x0100_0193;

/// `architecture.engram_indices`, evaluated for one position and one table.
///
/// Upstream expresses this as `acc ^= _shift_right(tokens, j)` over the whole
/// sequence, where `_shift_right(u, j)[t] == u[t - j]` (zero before the start).
/// Per position that is a hash over `tokens[t], tokens[t-1], ..., tokens[t-order+1]`.
///
/// `token_at(k)` must return the token `k` steps back from the current position,
/// or `0` where that falls before the window — which is what upstream's
/// zero-padding supplies.
#[inline]
pub fn engram_index<F: Fn(usize) -> u32>(
    table: usize,
    order: usize,
    slots: usize,
    token_at: F,
) -> usize {
    let seed = ENGRAM_SEED.wrapping_mul(table as u32 + 1);
    let mut acc = seed;
    for j in 0..order {
        acc = (acc ^ token_at(j)).wrapping_mul(ENGRAM_PRIME);
    }
    acc ^= acc >> 15;
    (acc % slots as u32) as usize
}

/// Which n-gram order table `table` belongs to, and its index within that order.
///
/// `engram_indices` iterates `for oi, order in enumerate(orders): for h in range(heads)`,
/// so tables are order-major.
#[inline]
pub fn engram_table_order(table: usize, heads: usize) -> usize {
    debug_assert!(heads > 0);
    table / heads
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn rms_unit_normalises_to_unit_rms() {
        let mut x = vec![3.0f32, -4.0, 0.0, 5.0];
        rms_unit(&mut x);
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        assert!((mean_sq - 1.0).abs() < 1e-5, "mean_sq={mean_sq}");
        // Direction preserved.
        assert!(x[0] > 0.0 && x[1] < 0.0 && x[2] == 0.0);
    }

    #[test]
    fn rms_unit_to_matches_in_place() {
        let x = vec![1.5f32, -2.5, 0.25, 7.0, -0.125];
        let mut a = x.clone();
        rms_unit(&mut a);
        let mut b = vec![0.0f32; x.len()];
        rms_unit_to(&x, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn rms_unit_handles_all_zeros() {
        let mut x = vec![0.0f32; 8];
        rms_unit(&mut x);
        assert!(x.iter().all(|v| *v == 0.0), "eps must keep this finite");
    }

    /// Sinkhorn's fixed point is doubly stochastic. Twenty iterations is enough
    /// on the 4x4 the model uses that both margins should be close to one.
    #[test]
    fn sinkhorn_output_is_near_doubly_stochastic() {
        let n = 4;
        let mut m: Vec<f32> = (0..n * n).map(|k| (k as f32 * 0.7).sin() * 2.0).collect();
        sinkhorn(&mut m, n);
        assert!(m.iter().all(|v| *v > 0.0 && *v < 1.0));
        for i in 0..n {
            let row: f32 = (0..n).map(|j| m[i * n + j]).sum();
            assert!((row - 1.0).abs() < 1e-3, "row {i} sums to {row}");
        }
        for j in 0..n {
            let col: f32 = (0..n).map(|i| m[i * n + j]).sum();
            assert!((col - 1.0).abs() < 1e-3, "col {j} sums to {col}");
        }
    }

    /// The reference initialises `b_res` to `4 * I`, which should drive Sinkhorn
    /// close to the identity — the lane-preserving residual.
    #[test]
    fn sinkhorn_of_scaled_identity_is_near_identity() {
        let n = 4;
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 4.0;
        }
        sinkhorn(&mut m, n);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (m[i * n + j] - want).abs() < 0.1,
                    "({i},{j}) = {} want ~{want}",
                    m[i * n + j]
                );
            }
        }
    }

    /// Row-major orientation must be right: a matrix favouring j = (i+1) % n
    /// must produce a permutation matrix with mass off the diagonal in that
    /// direction, not its transpose.
    #[test]
    fn sinkhorn_preserves_orientation() {
        let n = 4;
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + (i + 1) % n] = 6.0;
        }
        sinkhorn(&mut m, n);
        for i in 0..n {
            let on = m[i * n + (i + 1) % n];
            let diag = m[i * n + i];
            assert!(on > 0.9, "row {i}: expected mass at (i+1), got {on}");
            assert!(diag < 0.1, "row {i}: unexpected diagonal mass {diag}");
        }
    }

    #[test]
    fn silu_matches_definition() {
        for &x in &[-4.0f32, -1.0, 0.0, 0.5, 3.0] {
            let want = x / (1.0 + math::exp(-x));
            assert!((silu(x) - want).abs() < 1e-6, "x={x}");
        }
        assert_eq!(silu(0.0), 0.0);
    }

    /// Compare against a literal transcription of `decode._hadamard` using dense
    /// Walsh matmuls, so the FWHT substitution is checked in context.
    #[test]
    fn hadamard_mlp_matches_dense_reference() {
        for &(d, n) in &[(8usize, 8usize), (6, 8), (512, 512)] {
            let d1: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.03).sin()).collect();
            let d2: Vec<f32> = (0..n).map(|i| 0.9 + (i as f32 * 0.05).cos()).collect();
            let d3: Vec<f32> = (0..n).map(|i| 0.02 + i as f32 * 1e-4).collect();
            let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.11).sin()).collect();

            let mut scratch = vec![0.0f32; n];
            let mut got = vec![0.0f32; d];
            hadamard_mlp(&x, &d1, &d2, &d3, &mut scratch, &mut got);

            // Dense reference.
            let h = walsh_normalized(n);
            let mut z: Vec<f32> = (0..n).map(|i| if i < d { x[i] } else { 0.0 }).collect();
            for i in 0..n {
                z[i] *= d1[i];
            }
            z = matvec_row(&z, &h, n);
            for i in 0..n {
                z[i] = silu(z[i] * d2[i]);
            }
            z = matvec_row(&z, &h, n);
            for i in 0..d {
                let want = z[i] * d3[i];
                assert!(
                    (got[i] - want).abs() < 2e-5 * want.abs().max(1.0),
                    "d={d} n={n} i={i}: {} vs {want}",
                    got[i]
                );
            }
        }
    }

    fn walsh_normalized(n: usize) -> Vec<f32> {
        let mut h = vec![1.0f32];
        let mut size = 1;
        while size < n {
            let mut next = vec![0.0f32; size * 2 * size * 2];
            for r in 0..size {
                for c in 0..size {
                    let v = h[r * size + c];
                    next[r * size * 2 + c] = v;
                    next[r * size * 2 + c + size] = v;
                    next[(r + size) * size * 2 + c] = v;
                    next[(r + size) * size * 2 + c + size] = -v;
                }
            }
            h = next;
            size *= 2;
        }
        let s = 1.0 / math::sqrt(n as f32);
        h.iter().map(|v| v * s).collect()
    }

    /// Row-vector times matrix: `out[j] = sum_k z[k] * m[k * n + j]`.
    fn matvec_row(z: &[f32], m: &[f32], n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| (0..n).map(|k| z[k] * m[k * n + j]).sum())
            .collect()
    }

    /// Transcription of `engram_indices` for one position, to check the hash and
    /// the order-major table numbering.
    #[test]
    fn engram_index_matches_reference_hash() {
        let orders = [2usize, 3];
        let heads = 2usize;
        let slots = 8192usize;
        let tokens: [u32; 6] = [11, 250, 7, 4095, 1, 63];
        // Current position is the last token; step k back means tokens[len-1-k].
        let token_at = |k: usize| -> u32 {
            tokens
                .get(tokens.len().wrapping_sub(1 + k))
                .copied()
                .unwrap_or(0)
        };

        let mut table = 0usize;
        for (oi, &order) in orders.iter().enumerate() {
            for h in 0..heads {
                assert_eq!(engram_table_order(table, heads), oi, "table {table}");
                // Reference: seed = SEED * (oi*heads + h + 1)
                let seed = ENGRAM_SEED.wrapping_mul((oi * heads + h) as u32 + 1);
                let mut acc = seed;
                for j in 0..order {
                    acc = (acc ^ token_at(j)).wrapping_mul(ENGRAM_PRIME);
                }
                acc ^= acc >> 15;
                let want = (acc % slots as u32) as usize;
                assert_eq!(
                    engram_index(table, order, slots, token_at),
                    want,
                    "table {table}"
                );
                table += 1;
            }
        }
        assert_eq!(table, orders.len() * heads);
    }

    /// Different orders must give different indices for the same position —
    /// otherwise the two order groups would be redundant.
    #[test]
    fn engram_index_varies_with_order_and_table() {
        let tokens = [5u32, 9, 13];
        let at = |k: usize| {
            tokens
                .get(tokens.len().wrapping_sub(1 + k))
                .copied()
                .unwrap_or(0)
        };
        let a = engram_index(0, 2, 8192, at);
        let b = engram_index(0, 3, 8192, at);
        let c = engram_index(1, 2, 8192, at);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    /// Positions before the window start read as token 0, which is what
    /// upstream's `_shift_right` zero-padding supplies.
    #[test]
    fn engram_index_treats_missing_history_as_zero() {
        let at_short = |k: usize| if k == 0 { 42u32 } else { 0 };
        let tokens = [0u32, 42];
        let at_padded = |k: usize| tokens[tokens.len() - 1 - k];
        assert_eq!(
            engram_index(0, 2, 8192, at_short),
            engram_index(0, 2, 8192, at_padded)
        );
    }
}

/// Geometry-free primitives now shared with v3. Re-exported so existing v2
/// call sites keep their import path.
pub use crate::kernels::{rms_unit, rms_unit_to, sinkhorn, EPS, SINKHORN_ITERS};
