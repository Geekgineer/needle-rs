//! ZCRMSNorm: zero-centred RMSNorm — (1 + γ) * x / RMS(x).
//! Scale γ initialized to 0, so on first forward pass this reduces to plain RMSNorm.

use crate::math;

const EPS: f32 = 1e-6;

/// Apply ZCRMSNorm in-place.
/// `x`: shape [seq_len, d_model]
/// `scale`: shape [d_model]  (the γ parameter)
pub fn zc_rms_norm(x: &mut [f32], scale: &[f32], d_model: usize) {
    let seq_len = x.len() / d_model;
    debug_assert_eq!(x.len(), seq_len * d_model);
    debug_assert_eq!(scale.len(), d_model);

    for t in 0..seq_len {
        let row = &mut x[t * d_model..(t + 1) * d_model];
        zc_rms_norm_vec(row, scale);
    }
}

/// Apply ZCRMSNorm to a single vector in-place.
pub fn zc_rms_norm_vec(x: &mut [f32], scale: &[f32]) {
    debug_assert_eq!(x.len(), scale.len());
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / math::sqrt(mean_sq + EPS); // eps inside sqrt — matches Python
    for (xi, &si) in x.iter_mut().zip(scale.iter()) {
        *xi = (1.0 + si) * (*xi) * inv;
    }
}
