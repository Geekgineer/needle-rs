//! INT4 weight quantization/dequantization, bit-for-bit parity with Python quantize.py.
//!
//! Scheme: symmetric group-wise INT4, group_size=32 along input axis.
//! Scale = max(|group|) / 7.0, clamped >= 1e-8.
//! Values packed as two nibbles per byte: low nibble = even row, high nibble = odd row.
//! Stored signed as two's-complement 4-bit: range [-8, 7].
//!
//! Data layout: **row-major** (pair-major) — `data[pair * out_feat + o]`.
//! This makes the inner loop over output features contiguous, enabling AVX2
//! auto-vectorization in `matvec_inner_avx2` and plain scalar vectorization elsewhere.

use alloc::vec::Vec;
use alloc::vec;

pub const GROUP_SIZE: usize = 32;
pub const SCALE_MIN: f32 = 1e-8;

/// Packed INT4 weight tensor with per-group f32 scales.
///
/// `data`: row-major packed nibbles `[pair * out_feat + o]`,
///         total `(in_padded / 2) * out_feat` bytes.
/// `scales`: `[num_groups * out_feat]` f32, row-major `[g * out_feat + o]`.
/// `in_feat`: original (unpadded) input dimension.
/// `out_feat`: output dimension.
pub struct QuantizedWeight {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub in_feat: usize,
    pub out_feat: usize,
    pub num_groups: usize,
}

impl QuantizedWeight {
    /// Quantize an f32 weight matrix `w` of shape `[in_feat, out_feat]`.
    /// Matches Python `_fake_quantize_int4` exactly.
    pub fn quantize(w: &[f32], in_feat: usize, out_feat: usize) -> Self {
        assert_eq!(w.len(), in_feat * out_feat);

        let gs = GROUP_SIZE.min(in_feat);
        let pad = (gs - in_feat % gs) % gs;
        let in_padded = in_feat + pad;
        let num_groups = in_padded / gs;
        let num_pairs = in_padded / 2;

        // scales[g * out_feat + o] = max(|group|) / 7.0, clamped >= SCALE_MIN
        let mut scales = vec![0.0f32; num_groups * out_feat];
        for g in 0..num_groups {
            let row_start = g * gs;
            for o in 0..out_feat {
                let mut max_abs = 0.0f32;
                for r in 0..gs {
                    let global_row = row_start + r;
                    let val = if global_row < in_feat {
                        w[global_row * out_feat + o].abs()
                    } else {
                        0.0
                    };
                    if val > max_abs {
                        max_abs = val;
                    }
                }
                scales[g * out_feat + o] = (max_abs / 7.0).max(SCALE_MIN);
            }
        }

        // Pack nibbles in row-major order: data[pair * out_feat + o]
        // For pair p: r0 = 2*p, r1 = 2*p+1
        // byte = lo_nibble(w_q[r0, o]) | hi_nibble(w_q[r1, o])
        let mut data = vec![0u8; num_pairs * out_feat];
        for pair in 0..num_pairs {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let g = r0 / gs; // both r0, r1 are in the same group (gs is even)
            for o in 0..out_feat {
                let scale = scales[g * out_feat + o];
                let v0 = if r0 < in_feat { w[r0 * out_feat + o] } else { 0.0 };
                let v1 = if r1 < in_feat { w[r1 * out_feat + o] } else { 0.0 };
                let q0 = crate::math::round(v0 / scale).clamp(-8.0, 7.0) as i8;
                let q1 = crate::math::round(v1 / scale).clamp(-8.0, 7.0) as i8;
                let lo = (q0 as u8) & 0x0F;
                let hi = ((q1 as u8) & 0x0F) << 4;
                data[pair * out_feat + o] = lo | hi;
            }
        }

        Self { data, scales, in_feat, out_feat, num_groups }
    }

    /// Dequantize to f32 into `out` buffer of shape `[in_feat, out_feat]`.
    pub fn dequantize_to(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.in_feat * self.out_feat);

        let gs = GROUP_SIZE.min(self.in_feat);
        let num_pairs = self.num_groups * gs / 2;

        for pair in 0..num_pairs {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let g = r0 / gs;
            let base = pair * self.out_feat;
            for o in 0..self.out_feat {
                let byte = self.data[base + o];
                let lo = sign_extend4(byte & 0x0F);
                let hi = sign_extend4((byte >> 4) & 0x0F);
                let scale = self.scales[g * self.out_feat + o];
                if r0 < self.in_feat {
                    out[r0 * self.out_feat + o] = lo as f32 * scale;
                }
                if r1 < self.in_feat {
                    out[r1 * self.out_feat + o] = hi as f32 * scale;
                }
            }
        }
    }

    /// Matrix-vector multiply: y = W^T x  (y shape [out_feat], x shape [in_feat]).
    /// Dequantizes on the fly — never materializes the full weight matrix.
    /// Uses AVX2 auto-vectorized inner loop on x86_64 when the feature is present.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.in_feat);
        assert_eq!(y.len(), self.out_feat);

        for v in y.iter_mut() {
            *v = 0.0;
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd", target_feature = "avx2"))]
        // Safety: cfg ensures avx2 is present at compile time.
        return unsafe { self.matvec_avx2(x, y) };

        #[allow(unreachable_code)]
        self.matvec_scalar(x, y);
    }

    fn matvec_scalar(&self, x: &[f32], y: &mut [f32]) {
        let gs = GROUP_SIZE.min(self.in_feat);
        let num_pairs = self.num_groups * gs / 2;

        for pair in 0..num_pairs {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let g = r0 / gs;
            let x0 = if r0 < self.in_feat { x[r0] } else { 0.0 };
            let x1 = if r1 < self.in_feat { x[r1] } else { 0.0 };
            let base = pair * self.out_feat;
            let scale_base = g * self.out_feat;

            for o in 0..self.out_feat {
                let byte = self.data[base + o];
                let lo = sign_extend4(byte & 0x0F) as f32;
                let hi = sign_extend4((byte >> 4) & 0x0F) as f32;
                let scale = self.scales[scale_base + o];
                y[o] += (lo * x0 + hi * x1) * scale;
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn matvec_avx2(&self, x: &[f32], y: &mut [f32]) {
        use core::arch::x86_64::*;

        let gs = GROUP_SIZE.min(self.in_feat);
        let num_pairs = self.num_groups * gs / 2;
        let out_feat = self.out_feat;

        // AVX2: process 8 output features per SIMD lane.
        // For each pair: broadcast x0 and x1, then for each block of 8 output features:
        //   load 8 packed bytes, extract nibbles, sign-extend, convert to f32, FMA.
        let mask_lo = _mm256_set1_epi32(0x0F0F0F0F_u32 as i32);
        let bit3 = _mm256_set1_epi8(0x08_u8 as i8);
        let sign_fill = _mm256_set1_epi8(0xF0_u8 as i8);

        for pair in 0..num_pairs {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let g = r0 / gs;
            let x0 = if r0 < self.in_feat { x[r0] } else { 0.0 };
            let x1 = if r1 < self.in_feat { x[r1] } else { 0.0 };
            let x0v = _mm256_set1_ps(x0);
            let x1v = _mm256_set1_ps(x1);

            let data_base = pair * out_feat;
            let scale_base = g * out_feat;

            // Process 8 output features per iteration
            let full_blocks = out_feat / 8;
            let remainder = out_feat % 8;

            for blk in 0..full_blocks {
                let o = blk * 8;

                // Load 8 packed bytes (one per output feature for this pair)
                let bytes_128 = _mm_loadl_epi64(
                    self.data[data_base + o..].as_ptr() as *const __m128i
                );
                // Zero-extend to 256-bit (each byte in its own 32-bit lane for nicer ops)
                let bytes_256 = _mm256_cvtepu8_epi32(bytes_128);

                // Low nibbles: bits [3:0]
                let lo_256 = _mm256_and_si256(bytes_256, mask_lo);
                // High nibbles: shift right by 4, then mask
                let hi_256 = _mm256_and_si256(_mm256_srli_epi32(bytes_256, 4), mask_lo);

                // Sign-extend 4-bit → signed 32-bit:
                // If bit 3 set: value is negative → set bits [31:4] to all-1
                let lo_sign = _mm256_and_si256(lo_256, _mm256_set1_epi32(8));
                let lo_neg_mask = _mm256_cmpeq_epi32(lo_sign, _mm256_set1_epi32(8));
                let lo_ext = _mm256_and_si256(lo_neg_mask, _mm256_set1_epi32(-1i32 & !0x0F));
                let lo_i32 = _mm256_or_si256(lo_256, lo_ext);

                let hi_sign = _mm256_and_si256(hi_256, _mm256_set1_epi32(8));
                let hi_neg_mask = _mm256_cmpeq_epi32(hi_sign, _mm256_set1_epi32(8));
                let hi_ext = _mm256_and_si256(hi_neg_mask, _mm256_set1_epi32(-1i32 & !0x0F));
                let hi_i32 = _mm256_or_si256(hi_256, hi_ext);

                // Convert i32 → f32
                let lo_f32 = _mm256_cvtepi32_ps(lo_i32);
                let hi_f32 = _mm256_cvtepi32_ps(hi_i32);

                // Load scales for this block
                let scale_v = _mm256_loadu_ps(self.scales[scale_base + o..].as_ptr());

                // FMA: y[o..o+8] += (lo * x0 + hi * x1) * scale
                // Compute: contrib = (lo_f32 * x0v + hi_f32 * x1v) * scale_v
                let contrib_lo = _mm256_mul_ps(lo_f32, x0v);
                let contrib = _mm256_fmadd_ps(hi_f32, x1v, contrib_lo);
                let scaled = _mm256_mul_ps(contrib, scale_v);

                // Accumulate into y
                let y_ptr = y[o..].as_mut_ptr();
                let y_vec = _mm256_loadu_ps(y_ptr);
                let y_new = _mm256_add_ps(y_vec, scaled);
                _mm256_storeu_ps(y_ptr, y_new);
            }

            // Scalar tail for remaining output features (< 8)
            if remainder > 0 {
                let o_base = full_blocks * 8;
                let scale_off = scale_base + o_base;
                let data_off = data_base + o_base;
                for o in 0..remainder {
                    let byte = self.data[data_off + o];
                    let lo = sign_extend4(byte & 0x0F) as f32;
                    let hi = sign_extend4((byte >> 4) & 0x0F) as f32;
                    let scale = self.scales[scale_off + o];
                    y[o_base + o] += (lo * x0 + hi * x1) * scale;
                }
            }
        }

        // Suppress unused variable warnings for avx2-only variables when out_feat % 8 == 0
        let _ = mask_lo;
        let _ = bit3;
        let _ = sign_fill;
    }

    /// Batched matmul: Y = X W  (X shape [batch, in_feat], Y shape [batch, out_feat]).
    pub fn matmul(&self, x: &[f32], batch: usize, y: &mut [f32]) {
        assert_eq!(x.len(), batch * self.in_feat);
        assert_eq!(y.len(), batch * self.out_feat);
        for b in 0..batch {
            self.matvec(
                &x[b * self.in_feat..(b + 1) * self.in_feat],
                &mut y[b * self.out_feat..(b + 1) * self.out_feat],
            );
        }
    }
}

#[inline(always)]
fn sign_extend4(nibble: u8) -> i8 {
    if nibble & 0x8 != 0 {
        (nibble | 0xF0) as i8
    } else {
        nibble as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_extend() {
        assert_eq!(sign_extend4(0x7), 7);
        assert_eq!(sign_extend4(0x8), -8);
        assert_eq!(sign_extend4(0xF), -1);
        assert_eq!(sign_extend4(0x0), 0);
    }

    #[test]
    fn test_roundtrip_identity() {
        let in_feat = 64;
        let out_feat = 32;
        let mut w = alloc::vec![0.0f32; in_feat * out_feat];
        for i in 0..in_feat {
            for o in 0..out_feat {
                w[i * out_feat + o] = ((i + o) % 8) as f32;
            }
        }
        let qw = QuantizedWeight::quantize(&w, in_feat, out_feat);
        let mut out = alloc::vec![0.0f32; in_feat * out_feat];
        qw.dequantize_to(&mut out);

        for i in 0..w.len() {
            let diff = (w[i] - out[i]).abs();
            assert!(diff < 1e-5, "index {i}: expected {}, got {}", w[i], out[i]);
        }
    }

    #[test]
    fn test_matvec_matches_dequantize() {
        let in_feat = 64;
        let out_feat = 32;
        let w: alloc::vec::Vec<f32> = (0..in_feat * out_feat)
            .map(|i| (i as f32 - 32.0) * 0.05)
            .collect();
        let x: alloc::vec::Vec<f32> = (0..in_feat).map(|i| (i as f32 * 0.1).sin()).collect();

        let qw = QuantizedWeight::quantize(&w, in_feat, out_feat);

        // Reference: dequantize then naive matmul
        let mut dq = alloc::vec![0.0f32; in_feat * out_feat];
        qw.dequantize_to(&mut dq);
        let mut y_ref = alloc::vec![0.0f32; out_feat];
        for o in 0..out_feat {
            for i in 0..in_feat {
                y_ref[o] += x[i] * dq[i * out_feat + o];
            }
        }

        // matvec must produce the same result
        let mut y_got = alloc::vec![0.0f32; out_feat];
        qw.matvec(&x, &mut y_got);

        for o in 0..out_feat {
            let diff = (y_got[o] - y_ref[o]).abs();
            // Different accumulation order (per-pair vs per-row) causes ~machine-epsilon relative error.
            let tol = 1e-3 * (y_ref[o].abs().max(1.0));
            assert!(diff < tol, "out[{o}]: matvec={} dequant+dot={} diff={diff} tol={tol}", y_got[o], y_ref[o]);
        }
    }
}
