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

use alloc::vec;
use alloc::vec::Vec;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use core::sync::atomic::{AtomicU8, Ordering};

pub const GROUP_SIZE: usize = 32;
pub const SCALE_MIN: f32 = 1e-8;

// ─── Runtime AVX2 detection ───────────────────────────────────────────────────
// 0 = not yet checked, 1 = present, 2 = absent.
// AtomicU8 lives in core, so this is no_std compatible.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
static AVX2_DETECTED: AtomicU8 = AtomicU8::new(0);

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
pub fn has_avx2() -> bool {
    let cached = AVX2_DETECTED.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 1;
    }
    // Check CPUID leaf 7, subleaf 0: EBX bit 5 = AVX2.
    // CPUID is always available on x86_64.
    #[allow(unused_unsafe)]
    let result = unsafe {
        use core::arch::x86_64::__cpuid_count;
        let r = __cpuid_count(7, 0);
        (r.ebx >> 5) & 1 == 1
    };
    AVX2_DETECTED.store(if result { 1 } else { 2 }, Ordering::Relaxed);
    result
}

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
                let v0 = if r0 < in_feat {
                    w[r0 * out_feat + o]
                } else {
                    0.0
                };
                let v1 = if r1 < in_feat {
                    w[r1 * out_feat + o]
                } else {
                    0.0
                };
                let q0 = crate::math::round(v0 / scale).clamp(-8.0, 7.0) as i8;
                let q1 = crate::math::round(v1 / scale).clamp(-8.0, 7.0) as i8;
                let lo = (q0 as u8) & 0x0F;
                let hi = ((q1 as u8) & 0x0F) << 4;
                data[pair * out_feat + o] = lo | hi;
            }
        }

        Self {
            data,
            scales,
            in_feat,
            out_feat,
            num_groups,
        }
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

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if has_avx2() {
            // Safety: has_avx2() confirmed AVX2 via CPUID before this call.
            return unsafe { self.matvec_avx2(x, y) };
        }

        // NEON is mandatory on aarch64 (ARMv8 baseline) — no runtime check needed.
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return unsafe { self.matvec_neon(x, y) };
        }

        self.matvec_scalar(x, y);
    }

    #[allow(clippy::needless_range_loop)]
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
                let bytes_128 =
                    _mm_loadl_epi64(self.data[data_base + o..].as_ptr() as *const __m128i);
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
                let lo_ext = _mm256_and_si256(lo_neg_mask, _mm256_set1_epi32(!0x0F));
                let lo_i32 = _mm256_or_si256(lo_256, lo_ext);

                let hi_sign = _mm256_and_si256(hi_256, _mm256_set1_epi32(8));
                let hi_neg_mask = _mm256_cmpeq_epi32(hi_sign, _mm256_set1_epi32(8));
                let hi_ext = _mm256_and_si256(hi_neg_mask, _mm256_set1_epi32(!0x0F));
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
    }

    /// NEON-accelerated matvec for aarch64.
    ///
    /// Processes 8 output features per SIMD step using 64-bit NEON loads and two
    /// float32x4_t lanes. NEON is mandatory on ARMv8 (aarch64) — no runtime check needed.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    #[target_feature(enable = "neon")]
    unsafe fn matvec_neon(&self, x: &[f32], y: &mut [f32]) {
        use core::arch::aarch64::*;

        let gs = GROUP_SIZE.min(self.in_feat);
        let num_pairs = self.num_groups * gs / 2;
        let out_feat = self.out_feat;

        let mask_u8 = vdup_n_u8(0x0F);
        let eight_u8 = vdup_n_u8(8);
        let sixteen = vdup_n_u8(16);

        for pair in 0..num_pairs {
            let r0 = pair * 2;
            let r1 = pair * 2 + 1;
            let g = r0 / gs;
            let x0 = if r0 < self.in_feat {
                *x.get_unchecked(r0)
            } else {
                0.0
            };
            let x1 = if r1 < self.in_feat {
                *x.get_unchecked(r1)
            } else {
                0.0
            };
            let x0v = vdupq_n_f32(x0);
            let x1v = vdupq_n_f32(x1);

            let data_base = pair * out_feat;
            let scale_base = g * out_feat;

            // 8 output features per NEON step (one vld1_u8 = 8 packed bytes).
            let full8 = out_feat / 8;

            for blk in 0..full8 {
                let o = blk * 8;

                // Load 8 packed bytes: each byte encodes one pair of nibbles for one output feature.
                let bytes = vld1_u8(self.data.get_unchecked(data_base + o) as *const u8);

                // Low nibbles [3:0] and high nibbles [7:4] — values 0..15 as u8.
                let lo_raw = vand_u8(bytes, mask_u8);
                let hi_raw = vshr_n_u8::<4>(bytes); // logical shift, upper bits = 0

                // Sign-extend 4-bit → i8: values 8..15 become -8..-1 by subtracting 16.
                let lo_s8 = vreinterpret_s8_u8(vsub_u8(
                    lo_raw,
                    vand_u8(vcge_u8(lo_raw, eight_u8), sixteen),
                ));
                let hi_s8 = vreinterpret_s8_u8(vsub_u8(
                    hi_raw,
                    vand_u8(vcge_u8(hi_raw, eight_u8), sixteen),
                ));

                // Widen s8x8 → s16x8 → two s32x4 → two f32x4.
                let lo_s16 = vmovl_s8(lo_s8);
                let hi_s16 = vmovl_s8(hi_s8);

                // Lower 4 lanes (o..o+4)
                let lo_f32_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16)));
                let hi_f32_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16)));
                let scale_lo = vld1q_f32(self.scales.get_unchecked(scale_base + o) as *const f32);
                let contrib_lo = vmlaq_f32(vmulq_f32(lo_f32_lo, x0v), hi_f32_lo, x1v);
                let y_ptr_lo = y.get_unchecked_mut(o) as *mut f32;
                vst1q_f32(
                    y_ptr_lo,
                    vaddq_f32(vld1q_f32(y_ptr_lo), vmulq_f32(contrib_lo, scale_lo)),
                );

                // Upper 4 lanes (o+4..o+8)
                let lo_f32_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16)));
                let hi_f32_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16)));
                let scale_hi =
                    vld1q_f32(self.scales.get_unchecked(scale_base + o + 4) as *const f32);
                let contrib_hi = vmlaq_f32(vmulq_f32(lo_f32_hi, x0v), hi_f32_hi, x1v);
                let y_ptr_hi = y.get_unchecked_mut(o + 4) as *mut f32;
                vst1q_f32(
                    y_ptr_hi,
                    vaddq_f32(vld1q_f32(y_ptr_hi), vmulq_f32(contrib_hi, scale_hi)),
                );
            }

            // Scalar tail for remaining < 8 output features.
            let tail_start = full8 * 8;
            for o in tail_start..out_feat {
                let byte = *self.data.get_unchecked(data_base + o);
                let lo = sign_extend4(byte & 0x0F) as f32;
                let hi = sign_extend4((byte >> 4) & 0x0F) as f32;
                let scale = *self.scales.get_unchecked(scale_base + o);
                *y.get_unchecked_mut(o) += (lo * x0 + hi * x1) * scale;
            }
        }
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
            assert!(
                diff < tol,
                "out[{o}]: matvec={} dequant+dot={} diff={diff} tol={tol}",
                y_got[o],
                y_ref[o]
            );
        }
    }
}
