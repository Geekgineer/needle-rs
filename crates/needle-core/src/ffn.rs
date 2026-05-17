//! Feed-forward network (FFN) sub-layer.
//! Python: gate_proj + up_proj (both [d_model, d_ff]) → activation(gate)*up → down_proj [d_ff, d_model].
//! Activation variants: drelu = relu(gate)*relu(up), swiglu = silu(gate)*up, geglu = gelu(gate)*up.
//! In the default Needle config, encoder FFN is disabled (no_feedforward=true).

use crate::config::FfnActivation;
use crate::ops::{gelu, sigmoid};
use crate::quant::QuantizedWeight;
use alloc::vec;

pub struct FfnWeights {
    pub gate_proj: QuantizedWeight, // [d_model, d_ff]
    pub up_proj: QuantizedWeight,   // [d_model, d_ff]
    pub down_proj: QuantizedWeight, // [d_ff,    d_model]
}

/// FFN forward pass for a single token.
/// `out` must be length d_model; `d_ff` = gate_proj.out_feat.
pub fn ffn_forward(
    x: &[f32],
    out: &mut [f32],
    w: &FfnWeights,
    d_ff: usize,
    activation: &FfnActivation,
) {
    debug_assert_eq!(out.len(), x.len());

    let mut gate = vec![0.0f32; d_ff];
    let mut up = vec![0.0f32; d_ff];
    w.gate_proj.matvec(x, &mut gate);
    w.up_proj.matvec(x, &mut up);

    // Apply activation and fuse gate * up element-wise
    match activation {
        FfnActivation::SwiGLU => {
            for (g, u) in gate.iter_mut().zip(up.iter()) {
                // silu(g) = g * sigmoid(g)
                *g = *g * sigmoid(*g) * u;
            }
        }
        FfnActivation::GeGLU => {
            for (g, u) in gate.iter_mut().zip(up.iter()) {
                *g = gelu(*g) * u;
            }
        }
        FfnActivation::DRelu => {
            for (g, u) in gate.iter_mut().zip(up.iter()) {
                let rg = if *g > 0.0 { *g } else { 0.0 };
                let ru = if *u > 0.0 { *u } else { 0.0 };
                *g = rg * ru;
            }
        }
    }

    w.down_proj.matvec(&gate, out);
}
