//! Needle 3 mHC lane routing.
//!
//! The residual stream is `mhc_lanes` parallel copies of `d_model`. Each layer
//! reads all lanes, mixes them down to one vector for the block, and scatters
//! the block's delta back out under a learned doubly-stochastic mixing matrix.
//!
//! ```text
//! nx     = rms_unit(lanes flattened)                      // mhc_lanes * d_model
//! hpre   = sigmoid(a_pre · (nx · φ_pre)  + b_pre  + pre_off)
//! u      = Σ_i hpre[i] · lanes[i]                          // block input
//! y      = block(u) - u                                    // block delta
//! hpost  = 2 · sigmoid(a_post · (nx · φ_post) + b_post + post_off)
//! hres   = sinkhorn(a_res · (nx · φ_res) + b_res)          // lanes × lanes
//! lanes' = hres · lanes + hpost ⊗ y
//! ```
//!
//! The mathematics is the same as v2's. It is written separately because v2
//! expresses it as methods over its own state type, and restructuring a
//! token-exact path to share eighty lines is the worse trade.

extern crate alloc;

use crate::kernels::sinkhorn;
use crate::ops::sigmoid;

/// Per-layer mHC scalars and biases, already sliced to one layer.
pub struct MhcLayer<'a> {
    pub a_pre: f32,
    pub a_post: f32,
    pub a_res: f32,
    /// `[lanes]`.
    pub b_pre: &'a [f32],
    pub b_post: &'a [f32],
    /// `[lanes * lanes]`, row-major in `(i, j)`.
    pub b_res: &'a [f32],
}

/// `pre_off` — `8 · onehot(active lane) - 4`.
pub fn pre_off(layer: usize, lane: usize, lanes: usize) -> f32 {
    if lane == layer % lanes {
        4.0
    } else {
        -4.0
    }
}

/// `post_off` — `-4 · (1 - onehot(active lane))`.
pub fn post_off(layer: usize, lane: usize, lanes: usize) -> f32 {
    if lane == layer % lanes {
        0.0
    } else {
        -4.0
    }
}

/// Mix the lanes down to the block's input.
///
/// `phi_pre_rows` is `nx · φ_pre` for this layer — `lanes` values the caller
/// produced from whatever weight representation it holds. `hpre` is written
/// back as the gate actually used, so the scatter can reuse nothing and the
/// caller can trace it.
pub fn mix_down(
    lanes_buf: &[f32],
    phi_pre_rows: &mut [f32],
    m: &MhcLayer<'_>,
    layer: usize,
    lanes: usize,
    d_model: usize,
    u: &mut [f32],
) {
    debug_assert_eq!(lanes_buf.len(), lanes * d_model);
    debug_assert_eq!(phi_pre_rows.len(), lanes);
    debug_assert_eq!(u.len(), d_model);

    for (lane, h) in phi_pre_rows.iter_mut().enumerate() {
        *h = sigmoid(m.a_pre * *h + m.b_pre[lane] + pre_off(layer, lane, lanes));
    }
    for (c, uc) in u.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (lane, &h) in phi_pre_rows.iter().enumerate() {
            acc += h * lanes_buf[lane * d_model + c];
        }
        *uc = acc;
    }
}

/// Scatter the block delta back across the lanes.
///
/// `phi_post_rows` is `nx · φ_post` (`lanes` values) and `phi_res_rows` is
/// `nx · φ_res` (`lanes * lanes` values); both are overwritten in place with
/// the gates actually applied. `y` is `block(u) - u`.
pub fn scatter_up(
    lanes_buf: &mut [f32],
    phi_post_rows: &mut [f32],
    phi_res_rows: &mut [f32],
    y: &[f32],
    m: &MhcLayer<'_>,
    layer: usize,
    lanes: usize,
    d_model: usize,
    scratch: &mut [f32],
) {
    debug_assert_eq!(lanes_buf.len(), lanes * d_model);
    debug_assert_eq!(phi_post_rows.len(), lanes);
    debug_assert_eq!(phi_res_rows.len(), lanes * lanes);
    debug_assert_eq!(y.len(), d_model);
    debug_assert_eq!(scratch.len(), lanes * d_model);

    for (lane, h) in phi_post_rows.iter_mut().enumerate() {
        *h = 2.0 * sigmoid(m.a_post * *h + m.b_post[lane] + post_off(layer, lane, lanes));
    }
    for (idx, r) in phi_res_rows.iter_mut().enumerate() {
        *r = m.a_res * *r + m.b_res[idx];
    }
    sinkhorn(phi_res_rows, lanes);

    scratch.copy_from_slice(lanes_buf);
    for i in 0..lanes {
        let out = &mut lanes_buf[i * d_model..(i + 1) * d_model];
        let hp = phi_post_rows[i];
        for (c, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..lanes {
                acc += phi_res_rows[i * lanes + j] * scratch[j * d_model + c];
            }
            *o = acc + hp * y[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn offsets_single_out_the_active_lane() {
        // Layer 5 with 4 lanes is assigned lane 1.
        assert_eq!(pre_off(5, 1, 4), 4.0);
        assert_eq!(pre_off(5, 0, 4), -4.0);
        assert_eq!(post_off(5, 1, 4), 0.0);
        assert_eq!(post_off(5, 2, 4), -4.0);
    }

    #[test]
    fn mix_down_is_a_gated_lane_sum() {
        // Two lanes, d_model 2. With a_pre = 0 and b_pre chosen so the gates
        // are sigmoid(±4) the active lane dominates.
        let lanes_buf = [1.0f32, 2.0, 10.0, 20.0];
        let mut rows = [0.0f32, 0.0];
        let b_pre = [0.0f32, 0.0];
        let b_post = [0.0f32, 0.0];
        let b_res = [0.0f32; 4];
        let m = MhcLayer {
            a_pre: 0.0,
            a_post: 0.0,
            a_res: 0.0,
            b_pre: &b_pre,
            b_post: &b_post,
            b_res: &b_res,
        };
        let mut u = [0.0f32; 2];
        // layer 0 → active lane 0, so hpre = [sigmoid(4), sigmoid(-4)].
        mix_down(&lanes_buf, &mut rows, &m, 0, 2, 2, &mut u);
        let hi = sigmoid(4.0);
        let lo = sigmoid(-4.0);
        assert!((u[0] - (hi * 1.0 + lo * 10.0)).abs() < 1e-6, "{u:?}");
        assert!((u[1] - (hi * 2.0 + lo * 20.0)).abs() < 1e-6, "{u:?}");
    }

    #[test]
    fn scatter_rows_stay_doubly_stochastic() {
        // Sinkhorn makes hres rows and columns sum to one, so with y = 0 the
        // lane stream is a convex remix of itself and total mass is preserved.
        let lanes = 3usize;
        let d = 2usize;
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let before: f32 = buf.iter().sum();
        let mut post = vec![0.0f32; lanes];
        let mut res = vec![0.3f32; lanes * lanes];
        let b_pre = vec![0.0f32; lanes];
        let b_post = vec![-100.0f32; lanes]; // drive hpost to zero
        let b_res = vec![0.0f32; lanes * lanes];
        let m = MhcLayer {
            a_pre: 1.0,
            a_post: 1.0,
            a_res: 1.0,
            b_pre: &b_pre,
            b_post: &b_post,
            b_res: &b_res,
        };
        let y = vec![7.0f32, 8.0];
        let mut scratch = vec![0.0f32; lanes * d];
        scatter_up(
            &mut buf,
            &mut post,
            &mut res,
            &y,
            &m,
            0,
            lanes,
            d,
            &mut scratch,
        );
        let after: f32 = buf.iter().sum();
        assert!(
            (after - before).abs() < 1e-3,
            "mass changed: {before} -> {after}"
        );
    }
}
