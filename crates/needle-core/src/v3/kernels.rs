//! Needle 3 kernels that differ from v2.
//!
//! The important one is [`hadamard_mlp`]. Despite the name it is **not** a
//! Hadamard transform in v3: the three transform stages are Kronecker products
//! of learned 32x32 factors, initialised to Walsh matrices but trained away
//! from them (measured drift on the shipped weights is 1.45 to 3.98 against
//! entries of +/-1). So no fast Walsh transform applies, and the honest kernel
//! is a pair of small dense matmuls per stage.
//!
//! That is still the cheap option. At `n = 1024` a dense map would be about
//! 1.05M MACs per stage; the Kronecker factorisation costs
//! `2 * ba * bb * ba = 65,536`, a 16x saving, and 32x32 blocks stay in L1 and
//! vectorise cleanly — which is what makes v3 viable on constrained targets.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::math::exp;
use crate::ops::silu;

/// Kronecker block sizes for a transform width of `n`.
///
/// Mirrors upstream `_hada_blocks`: `ba = 1 << ((n - 1).bit_length() / 2)`,
/// `bb = n / ba`. At `n = 1024` both are 32.
pub fn hada_blocks(n: usize) -> (usize, usize) {
    debug_assert!(
        n.is_power_of_two(),
        "transform width must be a power of two"
    );
    let bits = usize::BITS - (n - 1).leading_zeros();
    let ba = 1usize << (bits / 2);
    (ba, n / ba)
}

/// Apply `out = aᵀ · Z · b`, where `z` is `Z` flattened row-major as
/// `(ba, bb)`.
///
/// This is upstream's `_kron_apply`, whose einsum is
/// `"...ij,ik,jl->...kl"` — that is, `out[k, l] = Σ_ij z[i, j] a[i, k] b[j, l]`.
pub fn kron_apply(z: &[f32], a: &[f32], b: &[f32], ba: usize, bb: usize, out: &mut [f32]) {
    debug_assert_eq!(z.len(), ba * bb);
    debug_assert_eq!(a.len(), ba * ba);
    debug_assert_eq!(b.len(), bb * bb);
    debug_assert_eq!(out.len(), ba * bb);

    // Stage one: t[k, j] = Σ_i z[i, j] · a[i, k]. Walking `i` in the outer
    // loop keeps `z`'s row and `a`'s row contiguous.
    let mut t = vec![0.0f32; ba * bb];
    for i in 0..ba {
        let zi = &z[i * bb..(i + 1) * bb];
        let ai = &a[i * ba..(i + 1) * ba];
        for (k, &aik) in ai.iter().enumerate() {
            if aik == 0.0 {
                continue;
            }
            let tk = &mut t[k * bb..(k + 1) * bb];
            for (tkj, &zij) in tk.iter_mut().zip(zi) {
                *tkj += zij * aik;
            }
        }
    }

    // Stage two: out[k, l] = Σ_j t[k, j] · b[j, l].
    for k in 0..ba {
        let tk = &t[k * bb..(k + 1) * bb];
        let ok = &mut out[k * bb..(k + 1) * bb];
        ok.fill(0.0);
        for (j, &tkj) in tk.iter().enumerate() {
            if tkj == 0.0 {
                continue;
            }
            let bj = &b[j * bb..(j + 1) * bb];
            for (okl, &bjl) in ok.iter_mut().zip(bj) {
                *okl += tkj * bjl;
            }
        }
    }
}

/// The learned factors and diagonals of one layer's HadamardMLP.
#[derive(Debug, Clone)]
pub struct HadaMlp {
    /// Diagonals, `hada_n` long each.
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub b2: Vec<f32>,
    pub d3: Vec<f32>,
    pub d4: Vec<f32>,
    /// Kronecker factor pairs for the three stages, `(ba*ba, bb*bb)` each.
    pub w1: (Vec<f32>, Vec<f32>),
    pub w2: (Vec<f32>, Vec<f32>),
    pub w3: (Vec<f32>, Vec<f32>),
    /// Low-rank conditioning: `[d_model, rank]` then `[rank, hada_n]`.
    pub cond_v: Vec<f32>,
    pub cond_u: Vec<f32>,
    pub cond_rank: usize,
}

/// Permutations applied after stages one and two. Stored in the container as
/// FP32 rather than recomputed, because upstream derives them from a seeded
/// RNG that a runtime has no way to reproduce.
#[derive(Debug, Clone)]
pub struct HadaPerms {
    pub p1: Vec<u32>,
    pub p2: Vec<u32>,
}

/// One position through HadamardMLP.
///
/// ```text
/// cond = 1 + softmax(x · cond_v) · cond_u
/// z    = pad(x, n)
/// z    = kron(d1 ⊙ z,                 w1)[p1]
/// z    = kron(silu(d2 ⊙ cond ⊙ z + b2), w2)[p2]
/// z    = kron(d3 ⊙ z,                 w3)
/// out  = (d4 ⊙ z)[..d_model]
/// ```
pub fn hadamard_mlp(x: &[f32], w: &HadaMlp, perms: &HadaPerms, hada_n: usize, out: &mut [f32]) {
    let d_model = x.len();
    debug_assert_eq!(out.len(), d_model);
    let (ba, bb) = hada_blocks(hada_n);

    // cond = 1 + softmax(x · cond_v) · cond_u
    let r = w.cond_rank;
    let mut proj = vec![0.0f32; r];
    for (j, p) in proj.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            acc += xi * w.cond_v[i * r + j];
        }
        *p = acc;
    }
    let max = proj.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in proj.iter_mut() {
        *p = exp(*p - max);
        sum += *p;
    }
    for p in proj.iter_mut() {
        *p /= sum;
    }
    let mut cond = vec![1.0f32; hada_n];
    for (j, &pj) in proj.iter().enumerate() {
        if pj == 0.0 {
            continue;
        }
        let row = &w.cond_u[j * hada_n..(j + 1) * hada_n];
        for (c, &u) in cond.iter_mut().zip(row) {
            *c += pj * u;
        }
    }

    // z = pad(x, n), then the three stages.
    let mut z = vec![0.0f32; hada_n];
    z[..d_model].copy_from_slice(x);

    let mut buf = vec![0.0f32; hada_n];

    for (zi, &d) in z.iter_mut().zip(&w.d1) {
        *zi *= d;
    }
    kron_apply(&z, &w.w1.0, &w.w1.1, ba, bb, &mut buf);
    permute(&buf, &perms.p1, &mut z);

    for i in 0..hada_n {
        z[i] = silu(w.d2[i] * cond[i] * z[i] + w.b2[i]);
    }
    kron_apply(&z, &w.w2.0, &w.w2.1, ba, bb, &mut buf);
    permute(&buf, &perms.p2, &mut z);

    for (zi, &d) in z.iter_mut().zip(&w.d3) {
        *zi *= d;
    }
    kron_apply(&z, &w.w3.0, &w.w3.1, ba, bb, &mut buf);

    for (o, (&b, &d)) in out.iter_mut().zip(buf.iter().zip(&w.d4)) {
        *o = b * d;
    }
}

/// `out[i] = src[perm[i]]`, upstream's `z[..., p]` gather.
fn permute(src: &[f32], perm: &[u32], out: &mut [f32]) {
    debug_assert_eq!(src.len(), perm.len());
    for (o, &p) in out.iter_mut().zip(perm) {
        *o = src[p as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_upstream() {
        assert_eq!(hada_blocks(1024), (32, 32));
        assert_eq!(hada_blocks(512), (16, 32));
        assert_eq!(hada_blocks(256), (16, 16));
    }

    #[test]
    fn kron_matches_a_transpose_z_b() {
        // 2x2 blocks, worked by hand: out = aᵀ · Z · b.
        let (ba, bb) = (2usize, 2usize);
        let z = [1.0f32, 2.0, 3.0, 4.0]; // Z = [[1,2],[3,4]]
        let a = [1.0f32, 2.0, 3.0, 4.0]; // A = [[1,2],[3,4]]
        let b = [0.0f32, 1.0, 1.0, 0.0]; // B swaps columns
        let mut out = [0.0f32; 4];
        kron_apply(&z, &a, &b, ba, bb, &mut out);
        // AᵀZ = [[1,3],[2,4]]ᵀ… computed explicitly:
        // (AᵀZ)[k][j] = Σ_i Z[i][j] A[i][k]
        //   k=0: [1*1+3*3, 2*1+4*3] = [10, 14]
        //   k=1: [1*2+3*4, 2*2+4*4] = [14, 20]
        // then · B swaps each row: [14,10] and [20,14].
        assert_eq!(out, [14.0, 10.0, 20.0, 14.0]);
    }

    #[test]
    fn permute_gathers() {
        let src = [10.0f32, 20.0, 30.0];
        let perm = [2u32, 0, 1];
        let mut out = [0.0f32; 3];
        permute(&src, &perm, &mut out);
        assert_eq!(out, [30.0, 10.0, 20.0]);
    }
}
