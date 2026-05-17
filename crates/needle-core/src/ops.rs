//! Primitive tensor operations: f32 only, no allocations.
use crate::math;

/// In-place softmax over a slice (stable via max subtraction).
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // All-−∞ input: exp(−∞ − (−∞)) = exp(NaN) = NaN. Fall back to uniform.
    if max == f32::NEG_INFINITY {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = math::exp(*v - max);
        sum += *v;
    }
    // sum > 0 is guaranteed (at least one exp(0) = 1.0 from the max element),
    // but guard against pathological subnormal underflow.
    if sum == 0.0 {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Element-wise sigmoid: σ(x) = 1 / (1 + e^-x).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + math::exp(-x))
}

/// GELU activation (approximate, matches JAX nn.gelu default).
/// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
pub fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2/π)
    const A: f32 = 0.044_715;
    let inner = C * (x + A * x * x * x);
    0.5 * x * (1.0 + math::tanh(inner))
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// y += alpha * x  (SAXPY)
#[inline]
pub fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Element-wise multiply: y[i] *= x[i]
#[inline]
pub fn elementwise_mul(x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi *= xi;
    }
}

/// Dense matmul: y[b,j] = Σ_i x[b,i] * w[i,j]
/// w shape [in_dim, out_dim], x shape [batch, in_dim], y shape [batch, out_dim].
pub fn matmul(x: &[f32], w: &[f32], y: &mut [f32], batch: usize, in_dim: usize, out_dim: usize) {
    debug_assert_eq!(x.len(), batch * in_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(y.len(), batch * out_dim);
    for b in 0..batch {
        let xb = &x[b * in_dim..(b + 1) * in_dim];
        let yb = &mut y[b * out_dim..(b + 1) * out_dim];
        for j in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += xb[i] * w[i * out_dim + j];
            }
            yb[j] = acc;
        }
    }
}

/// Add bias in-place: x[b, j] += bias[j]
pub fn add_bias(x: &mut [f32], bias: &[f32], batch: usize, out_dim: usize) {
    debug_assert_eq!(x.len(), batch * out_dim);
    debug_assert_eq!(bias.len(), out_dim);
    for b in 0..batch {
        for j in 0..out_dim {
            x[b * out_dim + j] += bias[j];
        }
    }
}
