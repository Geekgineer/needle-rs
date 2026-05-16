//! Rotary Position Embeddings (RoPE).
//! Applied to Q and K before the attention dot product.
//! Uses base=10000, interleaved complex rotation on pairs of head-dim elements.

use alloc::vec;
use alloc::vec::Vec;
use crate::math;

/// Precomputed cosine/sine tables for RoPE.
/// Shape: [max_len, head_dim/2]
pub struct RopeCache {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub max_len: usize,
    pub half_dim: usize,
}

impl RopeCache {
    pub fn new(max_len: usize, head_dim: usize, base: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_len * half_dim];
        let mut sin = vec![0.0f32; max_len * half_dim];

        for pos in 0..max_len {
            for i in 0..half_dim {
                let theta = (pos as f32) / math::powf(base, 2.0 * i as f32 / head_dim as f32);
                cos[pos * half_dim + i] = math::cos(theta);
                sin[pos * half_dim + i] = math::sin(theta);
            }
        }

        Self { cos, sin, max_len, half_dim }
    }

    /// Apply RoPE to a query/key tensor in-place.
    /// `x`: shape [seq_len, num_heads, head_dim]
    /// `offset`: position offset (for KV cache / decoder incremental decoding)
    pub fn apply(&self, x: &mut [f32], seq_len: usize, num_heads: usize, head_dim: usize, offset: usize) {
        debug_assert_eq!(x.len(), seq_len * num_heads * head_dim);
        debug_assert_eq!(head_dim, self.half_dim * 2);

        for t in 0..seq_len {
            let pos = t + offset;
            debug_assert!(pos < self.max_len, "RoPE position {} exceeds max_len {}", pos, self.max_len);
            let cos_row = &self.cos[pos * self.half_dim..(pos + 1) * self.half_dim];
            let sin_row = &self.sin[pos * self.half_dim..(pos + 1) * self.half_dim];

            for h in 0..num_heads {
                let base = (t * num_heads + h) * head_dim;
                let (left, right) = x[base..base + head_dim].split_at_mut(self.half_dim);
                for i in 0..self.half_dim {
                    let x0 = left[i];
                    let x1 = right[i];
                    left[i]  = x0 * cos_row[i] - x1 * sin_row[i];
                    right[i] = x1 * cos_row[i] + x0 * sin_row[i];
                }
            }
        }
    }
}
