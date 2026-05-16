//! Feed-forward network (FFN) sub-layer.
//! Python uses GELU, d_ff = 4 * d_model = 2048 by default.
//! In the default Needle config, encoder FFN is disabled (no_feedforward=true).
//! Decoder may also skip it — controlled by TransformerConfig.

use alloc::vec::Vec;
use alloc::vec;
use crate::ops::gelu;
use crate::quant::QuantizedWeight;

pub struct FfnWeights {
    pub w1: QuantizedWeight, // [d_model, d_ff]
    pub w2: QuantizedWeight, // [d_ff, d_model]
    pub b1: Vec<f32>,        // [d_ff]
    pub b2: Vec<f32>,        // [d_model]
}

/// FFN forward pass for a single token: out = W2(GELU(W1 x + b1)) + b2.
pub fn ffn_forward(x: &[f32], out: &mut [f32], w: &FfnWeights, d_ff: usize) {
    let d = x.len();
    debug_assert_eq!(out.len(), d);

    let mut hidden = vec![0.0f32; d_ff];
    w.w1.matvec(x, &mut hidden);
    if !w.b1.is_empty() {
        for (h, &b) in hidden.iter_mut().zip(w.b1.iter()) {
            *h += b;
        }
    }
    for h in hidden.iter_mut() {
        *h = gelu(*h);
    }
    w.w2.matvec(&hidden, out);
    if !w.b2.is_empty() {
        for (o, &b) in out.iter_mut().zip(w.b2.iter()) {
            *o += b;
        }
    }
}
