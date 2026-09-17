//! Needle 3 probe heads.
//!
//! v3 exports one: confidence. The contrastive head v2 carried is not in this
//! container (`extras.heads = ["confidence"]`), so `retrieve_tools` has no v3
//! equivalent — the same situation as the published v1 weights.
//!
//! A probe head reads the per-layer cells rather than the final hidden state,
//! and pools them twice: once over positions, per (layer, probe), and once
//! over (layer, probe), per query. Both poolings are softmaxes, so the head
//! sees the whole sequence rather than just the last token.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::kernels::rms_unit;
use crate::math::{exp, sqrt};

/// Head codes in `heads.manifest`, in upstream's canonical order.
pub const HEAD_EMBEDDING: u8 = 1;
pub const HEAD_CONFIDENCE: u8 = 2;
pub const HEAD_ROUTER: u8 = 3;

/// One probe head's weights, already dequantised.
///
/// Small enough that packing buys nothing: on the shipped model this is
/// 84x768 + 4x768 + a few vectors.
pub struct ProbeHead {
    /// `(cells * probes, d_model)`.
    pub probes: Vec<f32>,
    /// `(cells, probes)`.
    pub gain: Vec<f32>,
    /// `(queries, d_model)`.
    pub query: Vec<f32>,
    /// `(queries, cells, probes)`.
    pub row_bias: Vec<f32>,
    /// `(out_dim, queries * d_model)`.
    pub proj: Vec<f32>,
    /// `(out_dim)`.
    pub bias: Vec<f32>,
    pub cells: usize,
    pub probes_per_cell: usize,
    pub queries: usize,
    pub out_dim: usize,
}

impl ProbeHead {
    /// Run the head over `cells`, shaped `(seq, self.cells, d_model)`.
    ///
    /// Returns `out_dim` logits. Mirrors upstream `probe_pool` followed by
    /// `HeadProjection`.
    pub fn forward(&self, cells: &[f32], seq: usize, d: usize) -> Vec<f32> {
        debug_assert_eq!(cells.len(), seq * self.cells * d);
        let (l1, k, q) = (self.cells, self.probes_per_cell, self.queries);
        let scale = 1.0 / sqrt(d as f32);

        // r[l][k] — each (layer, probe) pair's softmax-weighted summary of the
        // sequence.
        let mut r = vec![0.0f32; l1 * k * d];
        let mut scores = vec![0.0f32; seq];
        for l in 0..l1 {
            for kk in 0..k {
                let pr = &self.probes[(l * k + kk) * d..(l * k + kk + 1) * d];
                let mut max = f32::NEG_INFINITY;
                for (t, sc) in scores.iter_mut().enumerate() {
                    let cell = &cells[(t * l1 + l) * d..(t * l1 + l + 1) * d];
                    let mut acc = 0.0f32;
                    for (a, b) in cell.iter().zip(pr) {
                        acc += a * b;
                    }
                    *sc = acc * scale;
                    if *sc > max {
                        max = *sc;
                    }
                }
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = exp(*sc - max);
                    sum += *sc;
                }
                let inv = 1.0 / sum;

                let dst = &mut r[(l * k + kk) * d..(l * k + kk + 1) * d];
                dst.fill(0.0);
                for (t, &sc) in scores.iter().enumerate() {
                    let w = sc * inv;
                    let cell = &cells[(t * l1 + l) * d..(t * l1 + l + 1) * d];
                    for (o, &c) in dst.iter_mut().zip(cell) {
                        *o += w * c;
                    }
                }
                // Normalise, then scale by this pair's learned gain.
                rms_unit(dst);
                let g = self.gain[l * k + kk];
                for o in dst.iter_mut() {
                    *o *= g;
                }
            }
        }

        // For each query, a softmax over all (layer, probe) pairs.
        let m = l1 * k;
        let mut pooled = vec![0.0f32; q * d];
        let mut u = vec![0.0f32; m];
        for qi in 0..q {
            let qv = &self.query[qi * d..(qi + 1) * d];
            let mut max = f32::NEG_INFINITY;
            for (idx, uu) in u.iter_mut().enumerate() {
                let rv = &r[idx * d..(idx + 1) * d];
                let mut acc = 0.0f32;
                for (a, b) in rv.iter().zip(qv) {
                    acc += a * b;
                }
                *uu = acc * scale + self.row_bias[qi * m + idx];
                if *uu > max {
                    max = *uu;
                }
            }
            let mut sum = 0.0f32;
            for uu in u.iter_mut() {
                *uu = exp(*uu - max);
                sum += *uu;
            }
            let inv = 1.0 / sum;

            let dst = &mut pooled[qi * d..(qi + 1) * d];
            dst.fill(0.0);
            for (idx, &uu) in u.iter().enumerate() {
                let w = uu * inv;
                let rv = &r[idx * d..(idx + 1) * d];
                for (o, &v) in dst.iter_mut().zip(rv) {
                    *o += w * v;
                }
            }
        }

        // Final projection over the flattened (queries * d_model) vector.
        let mut out = vec![0.0f32; self.out_dim];
        for (o, dst) in out.iter_mut().enumerate() {
            let row = &self.proj[o * q * d..(o + 1) * q * d];
            let mut acc = 0.0f32;
            for (a, b) in row.iter().zip(&pooled) {
                acc += a * b;
            }
            *dst = acc + self.bias.get(o).copied().unwrap_or(0.0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A head with one cell, one probe and one query, so the pooling collapses
    /// to something checkable by hand.
    fn trivial(d: usize) -> ProbeHead {
        ProbeHead {
            probes: vec![0.0; d],
            gain: vec![1.0],
            query: vec![0.0; d],
            row_bias: vec![0.0],
            proj: vec![1.0; d],
            bias: vec![0.5],
            cells: 1,
            probes_per_cell: 1,
            queries: 1,
            out_dim: 1,
        }
    }

    #[test]
    fn a_zero_probe_averages_the_sequence() {
        // With a zero probe every position scores 0, so the softmax is uniform
        // and r is the mean cell — then rms_unit normalises it.
        let d = 4;
        let h = trivial(d);
        let cells = vec![
            1.0f32, 0.0, 0.0, 0.0, // t=0
            3.0, 0.0, 0.0, 0.0, // t=1
        ];
        let out = h.forward(&cells, 2, d);
        // mean = [2,0,0,0]; rms over 4 dims = sqrt(4/4) = 1 → unit = [2,0,0,0]/1
        // proj is all ones, so the logit is the sum plus the bias.
        assert!(
            (out[0] - (2.0 / (4.0f32 / 4.0).sqrt() + 0.5)).abs() < 1e-5,
            "{out:?}"
        );
    }

    #[test]
    fn the_gain_scales_the_summary() {
        let d = 4;
        let mut h = trivial(d);
        let cells = vec![2.0f32, 0.0, 0.0, 0.0];
        let base = h.forward(&cells, 1, d)[0] - 0.5;
        h.gain[0] = 3.0;
        let scaled = h.forward(&cells, 1, d)[0] - 0.5;
        assert!((scaled - 3.0 * base).abs() < 1e-4, "{scaled} vs 3 * {base}");
    }
}
