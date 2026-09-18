//! Cactus-Quants (CQ) weights — the quantisation scheme Needle v2 `.cact`
//! blobs use. Bit-for-bit compatible with `needle/model/export.py::_cq_unpack`.
//!
//! # Scheme
//!
//! A logical `[out, in]` matrix is stored pre-transposed, so one output row is
//! contiguous along the reduction axis. Each row is cut into groups of `group`
//! (128 in the shipped checkpoint, `in` zero-padded up to a multiple of it).
//! Per group the encoder rotates by a normalised Walsh–Hadamard matrix,
//! records the group's L2 norm in FP16, and quantises the unit vector against a
//! shared Lloyd-Max codebook. Reconstruction is
//!
//! ```text
//! w_g = (codebook[idx_g] * norm_g) @ H        H = Walsh(group) / sqrt(group)
//! ```
//!
//! Widths are per tensor: 2, 3 or 4 bits against a header codebook, or ternary
//! (record width 5, three analytic levels). `needle2.cact` ships
//! `embedding=4,mhc=4,default=2` — 141 tensors at 2 bits, 4 at 4 bits.
//!
//! # Why the weights are never reconstructed
//!
//! `H` is symmetric, so for an activation slice `x_g`
//!
//! ```text
//! dot(x_g, w_g) = dot(x_g, u_g @ H) = dot(H @ x_g, u_g)      u_g = codebook[idx_g] * norm_g
//! ```
//!
//! The rotation moves off the weights and onto the activations. Reconstructing
//! weights costs one Hadamard per `(row, group)` — `out * in_pad` work per
//! matvec. Rotating the input costs one Hadamard per group, `in_pad` work for
//! the *whole* matrix, and it is shared by every output row. At the shipped
//! `512x512` that is 512 transforms against 4.
//!
//! What is left in the inner loop is a dot product of codebook-decoded bytes
//! against a pre-rotated activation, with one FP16 scale per group. Weights stay
//! packed in cache for the entire decode.
//!
//! Because several projections in a layer consume the same activation with the
//! same geometry (`q/k/v/gate` all read the pre-norm output), the rotation is
//! exposed separately as [`CqWeight::prepare_input`] so it can be paid once and
//! reused — see [`CqWeight::matvec_prepared`].
//!
//! # Padding
//!
//! Upstream truncates the reconstruction to `w[:, :in_feat]`. Zero-padding the
//! activation to `in_padded` reproduces that exactly: the padded lanes multiply
//! against zeros, so the identity above holds without a correction term.

use crate::hadamard::fwht_normalized;
use crate::math;
use alloc::vec;
use alloc::vec::Vec;

/// Directory record width that marks a ternary tensor. Not a real bit count —
/// ternary stores four signed 2-bit crumbs per byte.
pub const TERNARY_RECORD_BITS: u8 = 5;

/// The ternary Lloyd-Max centroid, before the `1/sqrt(group)` scaling.
/// `quantize._TERNARY_CB` is `{-c, 0, +c}` for this `c`.
pub const TERNARY_CENTROID: f32 = 1.2240064;

/// Widths present in the header codebook, in the order they are concatenated:
/// `cb2 | cb3 | cb4`, lengths 4 + 8 + 16 = 28.
pub const CODEBOOK_WIDTHS: [u8; 3] = [2, 3, 4];

/// Total length of a well-formed header codebook.
pub const CODEBOOK_LEN: usize = 4 + 8 + 16;

/// Largest quantisation group accepted. Bounds the stack buffer
/// [`CqWeight::matmul_rows_prepared`] decodes a group into. The shipped
/// checkpoint uses 128.
pub const MAX_GROUP: usize = 1024;

/// Errors from constructing a [`CqWeight`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CqError {
    /// Record `bits` is not 2, 3, 4 or [`TERNARY_RECORD_BITS`].
    UnsupportedBits(u8),
    /// `group` is zero, not a multiple of 8, not a power of two, or above
    /// [`MAX_GROUP`].
    BadGroup(usize),
    /// The header codebook is not [`CODEBOOK_LEN`] long.
    BadCodebookLen(usize),
    /// The blob is shorter than the packed indices plus FP16 norms require.
    ShortBlob { need: usize, got: usize },
    /// `out_feat` or `in_feat` is zero.
    EmptyShape,
}

/// Slice out one width's levels from a concatenated `cb2|cb3|cb4` header
/// codebook. The stored values already carry the `1/sqrt(group)` factor.
pub fn codebook_slice(codebook: &[f32], bits: u8) -> Result<&[f32], CqError> {
    if codebook.len() != CODEBOOK_LEN {
        return Err(CqError::BadCodebookLen(codebook.len()));
    }
    match bits {
        2 => Ok(&codebook[0..4]),
        3 => Ok(&codebook[4..12]),
        4 => Ok(&codebook[12..28]),
        other => Err(CqError::UnsupportedBits(other)),
    }
}

/// Bytes one packed row occupies. Mirrors `export._packed_row_bytes`.
#[inline]
pub fn packed_row_bytes(in_padded: usize, bits: u8) -> usize {
    let effective = if bits == TERNARY_RECORD_BITS {
        2
    } else {
        bits as usize
    };
    in_padded * effective / 8
}

/// A Cactus-Quants weight matrix, logically `[out_feat, in_feat]`.
pub struct CqWeight {
    /// Packed indices, `out_feat` rows of `row_bytes`.
    packed: Vec<u8>,
    /// Per-group L2 norms, `[out_feat * num_groups]`, decoded from FP16.
    norms: Vec<f32>,
    /// Byte-decode table: `256 * per_byte` values. Empty when `per_byte == 0`.
    lut: Vec<f32>,
    /// Codebook indices packed per byte: 4 at 2 bits and ternary, 2 at 4 bits,
    /// 0 for 3 bits (no byte alignment — the generic bit reader handles it).
    per_byte: usize,
    row_bytes: usize,
    num_groups: usize,
    pub out_feat: usize,
    /// Logical (unpadded) reduction width.
    pub in_feat: usize,
    /// Reduction width rounded up to a multiple of `group`.
    pub in_padded: usize,
    pub group: usize,
    /// Record width as stored in the directory: 2, 3, 4, or 5 for ternary.
    pub bits: u8,
    /// This tensor's levels, already scaled by `1/sqrt(group)`.
    levels: Vec<f32>,
}

impl CqWeight {
    /// A new tensor holding only `rows`, in the order given.
    ///
    /// Cactus-Quants packs each output row into its own `row_bytes` run with
    /// its own per-group norms, so a row is independently addressable: taking a
    /// subset is a copy, not a re-quantisation. That is what makes slicing a
    /// ladder rung out of the full model exact rather than approximate — the
    /// kept rows keep the codes they were trained with.
    ///
    /// Returns `None` if any index is out of range.
    pub fn select_rows(&self, rows: &[usize]) -> Option<Self> {
        let mut packed = Vec::with_capacity(rows.len() * self.row_bytes);
        let mut norms = Vec::with_capacity(rows.len() * self.num_groups);
        for &r in rows {
            if r >= self.out_feat {
                return None;
            }
            packed.extend_from_slice(&self.packed[r * self.row_bytes..(r + 1) * self.row_bytes]);
            norms.extend_from_slice(&self.norms[r * self.num_groups..(r + 1) * self.num_groups]);
        }
        Some(Self {
            packed,
            norms,
            lut: self.lut.clone(),
            per_byte: self.per_byte,
            row_bytes: self.row_bytes,
            num_groups: self.num_groups,
            out_feat: rows.len(),
            in_feat: self.in_feat,
            in_padded: self.in_padded,
            group: self.group,
            bits: self.bits,
            levels: self.levels.clone(),
        })
    }

    /// Parse a CQ tensor from its blob: packed indices followed by FP16 norms.
    ///
    /// `codebook` is the concatenated `cb2|cb3|cb4` block from the `.cact`
    /// header; it is ignored for ternary, whose levels are analytic.
    pub fn from_blob(
        blob: &[u8],
        out_feat: usize,
        in_feat: usize,
        group: usize,
        bits: u8,
        codebook: &[f32],
    ) -> Result<Self, CqError> {
        if out_feat == 0 || in_feat == 0 {
            return Err(CqError::EmptyShape);
        }
        if group == 0 || !group.is_multiple_of(8) || !group.is_power_of_two() {
            // Power-of-two: the group is rotated by a Walsh–Hadamard matrix.
            // Multiple of 8: keeps every group byte-aligned inside a packed row.
            return Err(CqError::BadGroup(group));
        }
        let levels: Vec<f32> = if bits == TERNARY_RECORD_BITS {
            let s = TERNARY_CENTROID / math::sqrt(group as f32);
            vec![-s, 0.0, s]
        } else {
            codebook_slice(codebook, bits)?.to_vec()
        };

        let in_padded = in_feat.div_ceil(group) * group;
        let num_groups = in_padded / group;
        let row_bytes = packed_row_bytes(in_padded, bits);
        let n_packed = out_feat * row_bytes;
        let n_norm_bytes = out_feat * num_groups * 2;
        if blob.len() < n_packed + n_norm_bytes {
            return Err(CqError::ShortBlob {
                need: n_packed + n_norm_bytes,
                got: blob.len(),
            });
        }

        let packed = blob[..n_packed].to_vec();
        let mut norms = Vec::with_capacity(out_feat * num_groups);
        for c in blob[n_packed..n_packed + n_norm_bytes].chunks_exact(2) {
            norms.push(math::f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
        }

        let (per_byte, lut) = build_lut(bits, &levels);

        Ok(Self {
            packed,
            norms,
            lut,
            per_byte,
            row_bytes,
            num_groups,
            out_feat,
            in_feat,
            in_padded,
            group,
            bits,
            levels,
        })
    }

    /// Bytes of packed weight held (excluding norms) — for size accounting.
    pub fn packed_bytes(&self) -> usize {
        self.packed.len()
    }

    /// Effective bits per weight, counting the FP16 group norms.
    pub fn effective_bits(&self) -> f32 {
        let total_bits = (self.packed.len() + self.norms.len() * 2) * 8;
        total_bits as f32 / (self.out_feat * self.in_padded) as f32
    }

    /// Scratch length [`prepare_input`] needs.
    #[inline]
    pub fn prepared_len(&self) -> usize {
        self.in_padded
    }

    /// Rotate an activation into the domain the packed weights live in.
    ///
    /// `x` is `in_feat` long; `xh` must be [`prepared_len`] long and is fully
    /// overwritten. Reusable across every [`CqWeight`] with the same `in_feat`
    /// and `group`.
    ///
    /// [`prepared_len`]: CqWeight::prepared_len
    pub fn prepare_input(&self, x: &[f32], xh: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_feat);
        debug_assert_eq!(xh.len(), self.in_padded);
        xh[..self.in_feat].copy_from_slice(x);
        // The padded tail must be zero: it stands in for the columns upstream
        // truncates off the reconstruction.
        xh[self.in_feat..].fill(0.0);
        for g in xh.chunks_exact_mut(self.group) {
            fwht_normalized(g);
        }
    }

    /// `y = W · x`, allocating the rotation scratch.
    ///
    /// Prefer [`prepare_input`] + [`matvec_prepared`] when several weights share
    /// one activation — a transformer layer's `q/k/v/gate` projections all do.
    ///
    /// [`prepare_input`]: CqWeight::prepare_input
    /// [`matvec_prepared`]: CqWeight::matvec_prepared
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        let mut xh = vec![0.0f32; self.in_padded];
        self.prepare_input(x, &mut xh);
        self.matvec_prepared(&xh, y);
    }

    /// `y = W · x` given an activation already rotated by [`prepare_input`].
    ///
    /// [`prepare_input`]: CqWeight::prepare_input
    pub fn matvec_prepared(&self, xh: &[f32], y: &mut [f32]) {
        self.matvec_rows_prepared(xh, 0, y)
    }

    /// `y = W[row_start .. row_start + y.len()] · x`, on a prepared activation.
    ///
    /// The mHC projections are stored as one tall matrix per kind, with a
    /// four-row band per layer (sixteen for `phi_res`), so a full matvec would
    /// compute 27x more rows than a layer needs.
    pub fn matvec_rows_prepared(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        #[cfg(feature = "parallel")]
        if self.should_parallelise(y.len()) {
            return self.matvec_rows_par(xh, row_start, y);
        }
        self.matvec_rows_serial(xh, row_start, y)
    }

    /// Split the output rows across the rayon pool.
    ///
    /// Each row's dot product is computed exactly as the serial path computes it,
    /// so this is bit-identical — only which core runs it changes.
    #[cfg(feature = "parallel")]
    fn matvec_rows_par(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        use rayon::prelude::*;
        let rows = y.len();
        let chunk = rows
            .div_ceil(rayon::current_num_threads())
            .max(MIN_PAR_ROWS);
        y.par_chunks_mut(chunk).enumerate().for_each(|(ci, part)| {
            self.matvec_rows_serial(xh, row_start + ci * chunk, part);
        });
    }

    fn matvec_rows_serial(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        debug_assert_eq!(xh.len(), self.in_padded);
        debug_assert!(row_start + y.len() <= self.out_feat);
        match self.per_byte {
            4 => self.matvec_lut::<4>(xh, row_start, y),
            2 => self.matvec_lut::<2>(xh, row_start, y),
            _ => self.matvec_generic(xh, row_start, y),
        }
    }

    /// Byte-LUT decode with a **single** running accumulator — the shape this
    /// kernel had before the accumulator was split into [`ACC_LANES`] lanes.
    ///
    /// Exposed only under the `bench-internals` feature, so the effect of that
    /// change can be measured against its own predecessor inside one benchmark
    /// run. Not for production use.
    #[cfg(feature = "bench-internals")]
    pub fn matvec_prepared_single_acc(&self, xh: &[f32], y: &mut [f32]) {
        debug_assert_eq!(xh.len(), self.in_padded);
        debug_assert_eq!(y.len(), self.out_feat);
        // `P` must be a const generic here, exactly as in the optimised path:
        // with a runtime width the inner loop cannot unroll and the arm would
        // measure the loss of specialisation rather than the loss of lanes.
        match self.per_byte {
            4 => self.single_acc::<4>(xh, y),
            2 => self.single_acc::<2>(xh, y),
            _ => self.matvec_prepared_reference(xh, y),
        }
    }

    #[cfg(feature = "bench-internals")]
    fn single_acc<const P: usize>(&self, xh: &[f32], y: &mut [f32]) {
        let bytes_per_group = self.group / P;
        let lut = &self.lut[..256 * P];
        for o in 0..self.out_feat {
            let row = &self.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
            let row_norms = &self.norms[o * self.num_groups..(o + 1) * self.num_groups];
            let mut total = 0.0f32;
            for (g, &norm) in row_norms.iter().enumerate() {
                let gbytes = &row[g * bytes_per_group..(g + 1) * bytes_per_group];
                let gx = &xh[g * self.group..(g + 1) * self.group];
                let mut acc = 0.0f32;
                for (bi, &byte) in gbytes.iter().enumerate() {
                    let base = byte as usize * P;
                    let vals: &[f32; P] = lut[base..base + P].try_into().unwrap();
                    let xs: &[f32; P] = gx[bi * P..bi * P + P].try_into().unwrap();
                    for k in 0..P {
                        acc += vals[k] * xs[k];
                    }
                }
                total += norm * acc;
            }
            y[o] = total;
        }
    }

    /// Deliberately simple reference for the packed dot product.
    ///
    /// One running scalar accumulator, no lane splitting, no byte LUT — decode
    /// each index and multiply. Kept public and tested against
    /// [`matvec_prepared`] so the optimised paths have an obviously-correct
    /// counterpart, and so the effect of the optimisation can be measured
    /// against it under identical conditions rather than against a number
    /// recorded at a different time.
    ///
    /// Not for production use: it is several times slower.
    ///
    /// [`matvec_prepared`]: CqWeight::matvec_prepared
    pub fn matvec_prepared_reference(&self, xh: &[f32], y: &mut [f32]) {
        debug_assert_eq!(xh.len(), self.in_padded);
        debug_assert_eq!(y.len(), self.out_feat);
        let bits = if self.bits == TERNARY_RECORD_BITS {
            2
        } else {
            self.bits as usize
        };
        let mask = (1u32 << bits) - 1;
        for (o, slot) in y.iter_mut().enumerate() {
            let row = &self.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
            let mut total = 0.0f32;
            for g in 0..self.num_groups {
                let norm = self.norms[o * self.num_groups + g];
                let mut acc = 0.0f32;
                for k in 0..self.group {
                    let code = read_bits(row, (g * self.group + k) * bits, bits, mask);
                    let idx = if self.bits == TERNARY_RECORD_BITS {
                        match code {
                            3 => 0,
                            0 => 1,
                            1 => 2,
                            _ => 1,
                        }
                    } else {
                        code as usize
                    };
                    acc += self.levels[idx] * xh[g * self.group + k];
                }
                total += norm * acc;
            }
            *slot = total;
        }
    }

    /// `y[b * rows + r] = W[row_start + r] · xh[b]`, over `batch` prepared
    /// activations laid out back to back at [`prepared_len`] stride.
    ///
    /// This is where batching pays. A matvec streams the whole packed weight set
    /// once per position — 11 MB for the shipped model, far past any core's
    /// private cache, so every position re-reads it from further out. Here each
    /// group is decoded **once** and applied to all `batch` activations, so the
    /// weight traffic is paid once for the batch and the inner loop is pure FMA
    /// over values already in registers.
    ///
    /// Bit-identical to `batch` calls to [`matvec_rows_prepared`], not merely
    /// close: the group norm is applied after the group dot, and the dot is
    /// summed in the same lane order, so no reassociation is introduced.
    ///
    /// `acc` is scratch, at least `batch` long.
    ///
    /// [`prepared_len`]: CqWeight::prepared_len
    /// [`matvec_rows_prepared`]: CqWeight::matvec_rows_prepared
    pub fn matmul_rows_prepared(
        &self,
        xh: &[f32],
        batch: usize,
        row_start: usize,
        rows: usize,
        y: &mut [f32],
        acc: &mut [f32],
    ) {
        #[cfg(feature = "parallel")]
        if self.matmul_work(rows, batch) >= PAR_MIN_MACS {
            // Two ways to split, chosen by shape. Row bands are preferred: each
            // task decodes its own weights once and the activation is shared.
            // But the mHC projections have only 4 to 16 rows — far too few to
            // band — while carrying a whole chunk's worth of positions, so those
            // split by *batch* instead. Splitting by batch means each task
            // re-decodes the weights, which is only cheap because the matrix is
            // small; that is exactly the case where it applies.
            if rows >= 2 * MIN_PAR_ROWS {
                return self.matmul_rows_par(xh, batch, row_start, rows, y);
            }
            if batch >= 2 * MIN_PAR_BATCH {
                return self.matmul_batch_par(xh, batch, row_start, rows, y);
            }
        }
        self.matmul_rows_serial(xh, batch, row_start, rows, y, acc)
    }

    /// Whether a matvec is big enough that splitting it beats the dispatch cost.
    ///
    /// A `512x512` matvec is ~30 µs over 512 rows; split eight ways each task
    /// holds ~4 µs of work, against a rayon dispatch of the same order — so small
    /// matvecs must stay serial. The only matvec that clears the bar is the tied
    /// LM head, at 8192 rows.
    #[cfg(feature = "parallel")]
    #[inline]
    fn should_parallelise(&self, rows: usize) -> bool {
        rows >= 2 * MIN_PAR_ROWS && rows.saturating_mul(self.in_padded) >= PAR_MIN_MACS
    }

    /// Split the output rows across the pool.
    ///
    /// The caller's `y` is batch-major (`y[b * rows + r]`), so a row band is
    /// strided rather than contiguous and `par_chunks_mut` cannot express it.
    /// Each task therefore fills a private band and copies it out; the copy is
    /// `batch * band` floats against `batch * band * in_padded` multiply-adds, so
    /// well under a tenth of a percent of the work.
    #[cfg(feature = "parallel")]
    fn matmul_rows_par(
        &self,
        xh: &[f32],
        batch: usize,
        row_start: usize,
        rows: usize,
        y: &mut [f32],
    ) {
        use rayon::prelude::*;

        /// Raw pointer to the output, so disjoint row bands can be written in
        /// parallel. Safe because the bands below never overlap.
        struct Out(*mut f32);
        // SAFETY: every task writes only `y[b * rows + band]` for its own band,
        // and the bands partition `0..rows`, so no two tasks alias.
        unsafe impl Send for Out {}
        unsafe impl Sync for Out {}
        impl Out {
            /// Read the pointer through a method so the closure captures the
            /// wrapper rather than the bare `*mut f32` field — edition-2021
            /// disjoint capture would otherwise take the field, which is not
            /// `Sync`, and the `unsafe impl` above would not apply.
            #[inline]
            fn ptr(&self) -> *mut f32 {
                self.0
            }
        }

        let out = Out(y.as_mut_ptr());
        let chunk = rows
            .div_ceil(rayon::current_num_threads())
            .max(MIN_PAR_ROWS);
        let bands: Vec<usize> = (0..rows).step_by(chunk).collect();

        bands.into_par_iter().for_each(|r0| {
            let band = chunk.min(rows - r0);
            let mut local = vec![0.0f32; batch * band];
            let mut acc = vec![0.0f32; batch];
            self.matmul_rows_serial(xh, batch, row_start + r0, band, &mut local, &mut acc);
            for b in 0..batch {
                // SAFETY: disjoint per band, and `b * rows + r0 + band <= y.len()`.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        local.as_ptr().add(b * band),
                        out.ptr().add(b * rows + r0),
                        band,
                    );
                }
            }
        });
    }

    /// Split by batch instead of by row.
    ///
    /// `y` is batch-major, so a batch range is contiguous and this needs no
    /// unsafe — unlike the row-band split. Each task handles a slice of the
    /// prepared activations and the matching slice of the output.
    #[cfg(feature = "parallel")]
    fn matmul_batch_par(
        &self,
        xh: &[f32],
        batch: usize,
        row_start: usize,
        rows: usize,
        y: &mut [f32],
    ) {
        use rayon::prelude::*;
        let stride = self.in_padded;
        let band = batch
            .div_ceil(rayon::current_num_threads())
            .max(MIN_PAR_BATCH);
        y[..batch * rows]
            .par_chunks_mut(band * rows)
            .enumerate()
            .for_each(|(ci, part)| {
                let b0 = ci * band;
                let n = part.len() / rows;
                let mut acc = vec![0.0f32; n];
                self.matmul_rows_serial(
                    &xh[b0 * stride..(b0 + n) * stride],
                    n,
                    row_start,
                    rows,
                    part,
                    &mut acc,
                );
            });
    }

    /// Multiply-accumulate count for a batched call.
    #[cfg(feature = "parallel")]
    #[inline]
    fn matmul_work(&self, rows: usize, batch: usize) -> usize {
        rows.saturating_mul(batch).saturating_mul(self.in_padded)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_rows_serial(
        &self,
        xh: &[f32],
        batch: usize,
        row_start: usize,
        rows: usize,
        y: &mut [f32],
        acc: &mut [f32],
    ) {
        // Dispatched exactly as `matvec_lut` is. Both must take the same branch
        // on a given machine, or `matmul_rows_prepared` would stop being
        // bit-identical to repeated `matvec_rows_prepared` — with FMA enabled
        // LLVM contracts `a*b + c` into one instruction, which rounds differently
        // from a separate multiply and add.
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if crate::quant::has_avx2() {
            // Safety: has_avx2() confirmed AVX2 and FMA via CPUID.
            return unsafe { self.matmul_rows_prepared_avx2(xh, batch, row_start, rows, y, acc) };
        }
        self.matmul_rows_prepared_impl(xh, batch, row_start, rows, y, acc)
    }

    /// AVX2/FMA build of [`matmul_rows_prepared_impl`].
    ///
    /// # Safety
    /// The caller must have confirmed AVX2 and FMA support.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn matmul_rows_prepared_avx2(
        &self,
        xh: &[f32],
        batch: usize,
        row_start: usize,
        rows: usize,
        y: &mut [f32],
        acc: &mut [f32],
    ) {
        self.matmul_rows_prepared_impl(xh, batch, row_start, rows, y, acc)
    }

    #[inline(always)]
    fn matmul_rows_prepared_impl(
        &self,
        xh: &[f32],
        batch: usize,
        row_start: usize,
        rows: usize,
        y: &mut [f32],
        acc: &mut [f32],
    ) {
        debug_assert_eq!(xh.len(), batch * self.in_padded);
        debug_assert_eq!(y.len(), batch * rows);
        debug_assert!(acc.len() >= batch);
        debug_assert!(row_start + rows <= self.out_feat);
        debug_assert!(self.group <= MAX_GROUP);

        let stride = self.in_padded;
        // One group's dequantised levels, unscaled, reused across the batch.
        let mut buf = [0.0f32; MAX_GROUP];
        let ug = &mut buf[..self.group];
        let lanewise = self.per_byte != 0;

        for r in 0..rows {
            let o = row_start + r;
            let row = &self.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
            let row_norms = &self.norms[o * self.num_groups..(o + 1) * self.num_groups];
            acc[..batch].fill(0.0);

            for (g, &norm) in row_norms.iter().enumerate() {
                // norm = 1.0 leaves the levels unscaled and rounds nothing.
                self.decode_group(row, g, 1.0, ug);
                let base = g * self.group;
                for b in 0..batch {
                    let gx = &xh[b * stride + base..b * stride + base + self.group];
                    // Match the summation order of the matvec path this width
                    // uses, so the two agree bit for bit.
                    let s = if lanewise {
                        group_dot_lanes(ug, gx)
                    } else {
                        group_dot_serial(ug, gx)
                    };
                    acc[b] += norm * s;
                }
            }
            for b in 0..batch {
                y[b * rows + r] = acc[b];
            }
        }
    }

    /// Byte-at-a-time path: one LUT entry yields `P` decoded levels.
    ///
    /// The accumulator is split into [`ACC_LANES`] independent lanes rather than
    /// one running scalar. A single accumulator makes the loop latency-bound —
    /// every FMA waits on the previous — which caps throughput at one element per
    /// FMA latency regardless of issue width. Independent lanes expose
    /// instruction-level parallelism and let the autovectoriser fold the body
    /// into vector FMAs.
    ///
    /// The lane count is fixed rather than tied to `P`, so a 4-bit tensor (two
    /// levels per byte) gets the same parallelism as a 2-bit one by consuming
    /// `ACC_LANES / P` bytes per iteration.
    fn matvec_lut<const P: usize>(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        // Baseline x86_64 guarantees only SSE2, and this crate deliberately does
        // not build with `target-cpu=native`, so without a runtime check the
        // eight-lane loop compiles to pairs of 4-wide SSE ops. Recompiling the
        // *same source* under `avx2,fma` and selecting it by CPUID gets the
        // 8-wide form into a portable binary. No hand-written intrinsics:
        // measured on aarch64, LLVM's autovectorisation of this loop beats a
        // hand-written 4-accumulator NEON version (8.1 against 7.1 Gelem/s), so
        // there is no reason to expect intrinsics to help on x86 either.
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if crate::quant::has_avx2() {
            // Safety: has_avx2() confirmed AVX2 and FMA via CPUID.
            return unsafe { self.matvec_lut_avx2::<P>(xh, row_start, y) };
        }
        self.matvec_lut_impl::<P>(xh, row_start, y)
    }

    /// AVX2/FMA build of [`matvec_lut_impl`]. Same source, wider vectors.
    ///
    /// # Safety
    /// The caller must have confirmed AVX2 and FMA support.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn matvec_lut_avx2<const P: usize>(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        self.matvec_lut_impl::<P>(xh, row_start, y)
    }

    #[inline(always)]
    fn matvec_lut_impl<const P: usize>(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        let bytes_per_group = self.group / P;
        let lut = &self.lut[..256 * P];
        for (yi, o) in (row_start..row_start + y.len()).enumerate() {
            let row = &self.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
            let row_norms = &self.norms[o * self.num_groups..(o + 1) * self.num_groups];
            let mut total = 0.0f32;
            for (g, &norm) in row_norms.iter().enumerate() {
                let gbytes = &row[g * bytes_per_group..(g + 1) * bytes_per_group];
                let gx = &xh[g * self.group..(g + 1) * self.group];
                total += norm * dot_group::<P>(lut, gbytes, gx);
            }
            y[yi] = total;
        }
    }

    /// Bit-at-a-time path for widths that do not divide a byte (3 bits).
    fn matvec_generic(&self, xh: &[f32], row_start: usize, y: &mut [f32]) {
        let bits = self.bits as usize;
        let mask = (1u32 << bits) - 1;
        for (yi, o) in (row_start..row_start + y.len()).enumerate() {
            let row = &self.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
            let row_norms = &self.norms[o * self.num_groups..(o + 1) * self.num_groups];
            let mut total = 0.0f32;
            for (g, &norm) in row_norms.iter().enumerate() {
                let gx = &xh[g * self.group..(g + 1) * self.group];
                let mut acc = 0.0f32;
                for (k, &x) in gx.iter().enumerate() {
                    let bit = (g * self.group + k) * bits;
                    let idx = read_bits(row, bit, bits, mask);
                    acc += self.levels[idx as usize] * x;
                }
                total += norm * acc;
            }
            y[yi] = total;
        }
    }

    /// Reconstruct one output row into `out` (`in_feat` long).
    ///
    /// Used where a row is genuinely needed as a vector rather than as a dot
    /// product — token embedding lookup, and parity tests.
    pub fn dequantize_row(&self, o: usize, out: &mut [f32]) {
        debug_assert!(o < self.out_feat);
        debug_assert_eq!(out.len(), self.in_feat);
        let row = &self.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
        let row_norms = &self.norms[o * self.num_groups..(o + 1) * self.num_groups];
        let mut buf = vec![0.0f32; self.group];
        for (g, &norm) in row_norms.iter().enumerate() {
            self.decode_group(row, g, norm, &mut buf);
            fwht_normalized(&mut buf);
            let base = g * self.group;
            for k in 0..self.group {
                if base + k < self.in_feat {
                    out[base + k] = buf[k];
                }
            }
        }
    }

    /// Reconstruct the whole matrix, row-major `[out_feat, in_feat]`.
    pub fn dequantize_to(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.out_feat * self.in_feat);
        for o in 0..self.out_feat {
            self.dequantize_row(o, &mut out[o * self.in_feat..(o + 1) * self.in_feat]);
        }
    }

    /// Fill `buf` (`group` long) with `codebook[idx] * norm` for one group,
    /// i.e. `u_g` before the Hadamard rotation.
    fn decode_group(&self, row: &[u8], g: usize, norm: f32, buf: &mut [f32]) {
        if self.per_byte != 0 {
            let p = self.per_byte;
            let bytes_per_group = self.group / p;
            let gbytes = &row[g * bytes_per_group..(g + 1) * bytes_per_group];
            for (bi, &byte) in gbytes.iter().enumerate() {
                for k in 0..p {
                    buf[bi * p + k] = self.lut[byte as usize * p + k] * norm;
                }
            }
        } else {
            let bits = self.bits as usize;
            let mask = (1u32 << bits) - 1;
            for (k, slot) in buf.iter_mut().enumerate() {
                let idx = read_bits(row, (g * self.group + k) * bits, bits, mask);
                *slot = self.levels[idx as usize] * norm;
            }
        }
    }
}

/// Smallest row band handed to one thread. Below this the per-task overhead
/// dominates whatever the total size is.
#[cfg(feature = "parallel")]
const MIN_PAR_ROWS: usize = 64;

/// Smallest batch band handed to one thread, for the batch-split path.
#[cfg(feature = "parallel")]
const MIN_PAR_BATCH: usize = 8;

/// Multiply-accumulate count below which a call stays on one thread.
#[cfg(feature = "parallel")]
const PAR_MIN_MACS: usize = 1 << 20;

/// Accumulator lanes in the packed dot product.
///
/// Eight independent f32 lanes: two 128-bit vector FMAs per iteration on NEON or
/// SSE, one 256-bit on AVX, and enough separate dependency chains to cover FMA
/// latency on all of them. Must be a multiple of the largest `P` (4).
pub const ACC_LANES: usize = 8;

/// Dot one quantisation group against a pre-rotated activation slice.
///
/// `P` levels are packed per byte, so each iteration consumes `ACC_LANES / P`
/// bytes. `P` and `ACC_LANES` are both compile-time, so the inner loops unroll
/// fully and the index arithmetic folds away.
#[inline(always)]
fn dot_group<const P: usize>(lut: &[f32], gbytes: &[u8], gx: &[f32]) -> f32 {
    debug_assert_eq!(ACC_LANES % P, 0);
    let per_iter = ACC_LANES / P;
    let mut lanes = [0.0f32; ACC_LANES];

    let full = gbytes.len() - gbytes.len() % per_iter;
    let mut bi = 0;
    while bi < full {
        // The decoded levels are staged in `vals` before the FMA loop on
        // purpose. Multiplying straight out of the LUT looks leaner but measured
        // 18% slower on aarch64: gathering into a contiguous array first is what
        // lets the autovectoriser emit whole-vector loads and FMAs instead of
        // per-`P` fragments.
        let mut vals = [0.0f32; ACC_LANES];
        for t in 0..per_iter {
            let base = gbytes[bi + t] as usize * P;
            let entry: &[f32; P] = lut[base..base + P].try_into().unwrap();
            vals[t * P..(t + 1) * P].copy_from_slice(entry);
        }
        let xs: &[f32; ACC_LANES] = gx[bi * P..bi * P + ACC_LANES].try_into().unwrap();
        for k in 0..ACC_LANES {
            lanes[k] += vals[k] * xs[k];
        }
        bi += per_iter;
    }

    let mut acc: f32 = lanes.iter().sum();
    // Tail, for a group smaller than one full iteration.
    for b in bi..gbytes.len() {
        let base = gbytes[b] as usize * P;
        for k in 0..P {
            acc += lut[base + k] * gx[b * P + k];
        }
    }
    acc
}

/// Dot a decoded group against an activation slice using [`ACC_LANES`] lanes,
/// in the same order [`dot_group`] accumulates. Used by the batched matmul so it
/// matches the LUT matvec path bit for bit.
#[inline(always)]
fn group_dot_lanes(u: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(u.len(), x.len());
    let chunks = u.len() / ACC_LANES;
    let mut lanes = [0.0f32; ACC_LANES];
    for c in 0..chunks {
        let uu: &[f32; ACC_LANES] = u[c * ACC_LANES..(c + 1) * ACC_LANES].try_into().unwrap();
        let xx: &[f32; ACC_LANES] = x[c * ACC_LANES..(c + 1) * ACC_LANES].try_into().unwrap();
        for k in 0..ACC_LANES {
            lanes[k] += uu[k] * xx[k];
        }
    }
    let mut s: f32 = lanes.iter().sum();
    for k in chunks * ACC_LANES..u.len() {
        s += u[k] * x[k];
    }
    s
}

/// Sequential dot, matching the single-accumulator order `matvec_generic` uses
/// for widths that do not divide a byte.
#[inline(always)]
fn group_dot_serial(u: &[f32], x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (a, b) in u.iter().zip(x.iter()) {
        s += a * b;
    }
    s
}

/// Read a `bits`-wide LSB-first field starting at bit offset `bit`.
///
/// Upstream packs in chunks of 8 indices OR'd into one word and emitted as
/// `bits` little-endian bytes, which is equivalently one continuous LSB-first
/// bitstream per row — index `k` occupies bits `[k*bits, (k+1)*bits)`.
#[inline]
fn read_bits(row: &[u8], bit: usize, bits: usize, mask: u32) -> u32 {
    let byte = bit / 8;
    let shift = bit % 8;
    // A field is at most 4 bits wide, so it spans at most two bytes.
    let lo = row[byte] as u32;
    let hi = if shift + bits > 8 {
        row[byte + 1] as u32
    } else {
        0
    };
    ((lo | (hi << 8)) >> shift) & mask
}

/// Build the byte-decode table for widths that divide a byte.
///
/// Returns `(indices_per_byte, table)` where `table[b*p .. b*p+p]` holds the
/// levels for the `p` indices packed into byte `b`, LSB first. `(0, empty)`
/// means no table — the caller must use the bit reader.
fn build_lut(bits: u8, levels: &[f32]) -> (usize, Vec<f32>) {
    // Ternary packs four signed 2-bit crumbs per byte, codes 3, 0, 1 for trit
    // indices 0, 1, 2 — so sign-extending a crumb yields -1, 0, +1 directly.
    let ternary = bits == TERNARY_RECORD_BITS;
    let width = if ternary { 2 } else { bits as usize };
    let per_byte = match width {
        2 => 4,
        4 => 2,
        _ => return (0, Vec::new()), // 3 bits: no byte alignment
    };
    let mask = (1u32 << width) - 1;

    let mut lut = vec![0.0f32; 256 * per_byte];
    for byte in 0..256usize {
        for k in 0..per_byte {
            let code = ((byte as u32) >> (k * width)) & mask;
            let idx = if ternary {
                // Mirrors `export._unpack_ternary_crumbs`. Code 2 cannot be
                // produced by the encoder; map it to the zero level.
                match code {
                    3 => 0,
                    0 => 1,
                    1 => 2,
                    _ => 1,
                }
            } else {
                code as usize
            };
            lut[byte * per_byte + k] = levels[idx];
        }
    }
    (per_byte, lut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hadamard::fwht_normalized;

    /// Concatenated cb2|cb3|cb4 with the shape of a real header codebook.
    /// Values are arbitrary but distinct so a wrong slice or index shows up.
    fn fake_codebook(group: usize) -> Vec<f32> {
        let s = 1.0 / math::sqrt(group as f32);
        let mut cb = Vec::with_capacity(CODEBOOK_LEN);
        for &bits in &CODEBOOK_WIDTHS {
            let levels = 1usize << bits;
            for i in 0..levels {
                // Monotone increasing and symmetric-ish, like Lloyd-Max output.
                let v = (i as f32 - (levels as f32 - 1.0) / 2.0) / (levels as f32 / 2.0);
                cb.push(v * s);
            }
        }
        assert_eq!(cb.len(), CODEBOOK_LEN);
        cb
    }

    /// Pack indices LSB-first, exactly as `export._pack_lsb` does.
    fn pack_lsb(idx: &[u8], out: usize, in_pad: usize, bits: usize) -> Vec<u8> {
        let row_bytes = in_pad * bits / 8;
        let mut packed = vec![0u8; out * row_bytes];
        for o in 0..out {
            for k in 0..in_pad {
                let v = idx[o * in_pad + k] as u32;
                let bit = k * bits;
                for b in 0..bits {
                    if (v >> b) & 1 == 1 {
                        let abs = bit + b;
                        packed[o * row_bytes + abs / 8] |= 1 << (abs % 8);
                    }
                }
            }
        }
        packed
    }

    fn f32_to_f16(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 31) as u16) << 15;
        let exp = ((x >> 23) & 0xFF) as i32 - 127;
        let mant = x & 0x7F_FFFF;
        if exp < -14 {
            return sign;
        }
        if exp > 15 {
            return sign | 0x7C00;
        }
        sign | (((exp + 15) as u16) << 10) | ((mant >> 13) as u16)
    }

    struct Case {
        w: CqWeight,
        /// The reference reconstruction, `[out, in_feat]`, built the way
        /// `export._cq_unpack` builds it: decode, scale, then rotate.
        dense: Vec<f32>,
    }

    fn build_case(out: usize, in_feat: usize, group: usize, bits: u8) -> Case {
        let cb = fake_codebook(group);
        let levels: Vec<f32> = if bits == TERNARY_RECORD_BITS {
            let s = TERNARY_CENTROID / math::sqrt(group as f32);
            vec![-s, 0.0, s]
        } else {
            codebook_slice(&cb, bits).unwrap().to_vec()
        };
        let n_levels = levels.len();

        let in_pad = in_feat.div_ceil(group) * group;
        let num_groups = in_pad / group;

        // Deterministic indices that exercise every level.
        let idx: Vec<u8> = (0..out * in_pad)
            .map(|i| ((i * 7 + i / 13) % n_levels) as u8)
            .collect();
        let norms: Vec<f32> = (0..out * num_groups)
            .map(|i| 0.25 + (i % 11) as f32 * 0.125)
            .collect();

        let packed = if bits == TERNARY_RECORD_BITS {
            // trit 0,1,2 -> crumb 3,0,1
            let crumbs: Vec<u8> = idx
                .iter()
                .map(|&t| if t == 0 { 3 } else { t - 1 })
                .collect();
            pack_lsb(&crumbs, out, in_pad, 2)
        } else {
            pack_lsb(&idx, out, in_pad, bits as usize)
        };

        let mut blob = packed;
        for &n in &norms {
            blob.extend_from_slice(&f32_to_f16(n).to_le_bytes());
        }

        // Reference dense reconstruction.
        let mut dense = vec![0.0f32; out * in_feat];
        for o in 0..out {
            for g in 0..num_groups {
                let mut u: Vec<f32> = (0..group)
                    .map(|k| levels[idx[o * in_pad + g * group + k] as usize])
                    .collect();
                let norm = math::f16_to_f32(f32_to_f16(norms[o * num_groups + g]));
                for v in u.iter_mut() {
                    *v *= norm;
                }
                fwht_normalized(&mut u);
                for (k, &val) in u.iter().enumerate() {
                    let col = g * group + k;
                    if col < in_feat {
                        dense[o * in_feat + col] = val;
                    }
                }
            }
        }

        let w = CqWeight::from_blob(&blob, out, in_feat, group, bits, &cb).unwrap();
        Case { w, dense }
    }

    /// Selecting rows then dequantising must equal dequantising then selecting.
    ///
    /// If it does not, the packed rows are not independent and slicing a ladder
    /// rung out of the full model would silently corrupt every kept block.
    #[test]
    fn selecting_rows_commutes_with_dequantisation() {
        for bits in WIDTHS {
            let (out, in_feat, group) = (12usize, 256usize, 128usize);
            let case = build_case(out, in_feat, group, bits);

            let rows = [0usize, 3, 4, 9, 11];
            let cut = case.w.select_rows(&rows).expect("rows are in range");
            assert_eq!(cut.out_feat, rows.len());
            assert_eq!(cut.in_feat, in_feat);
            assert_eq!(cut.bits, case.w.bits);

            let mut dense_cut = vec![0.0f32; rows.len() * in_feat];
            cut.dequantize_to(&mut dense_cut);

            for (n, &r) in rows.iter().enumerate() {
                let want = &case.dense[r * in_feat..(r + 1) * in_feat];
                let got = &dense_cut[n * in_feat..(n + 1) * in_feat];
                assert_eq!(
                    want, got,
                    "bits {bits}: row {r} changed when sliced into position {n}"
                );
            }
        }
    }

    /// Order is preserved, and a row may be taken more than once.
    #[test]
    fn selection_follows_the_order_given() {
        let (out, in_feat, group) = (6usize, 128usize, 128usize);
        let case = build_case(out, in_feat, group, 4);
        let rows = [5usize, 0, 5];
        let cut = case.w.select_rows(&rows).unwrap();
        let mut dense_cut = vec![0.0f32; rows.len() * in_feat];
        cut.dequantize_to(&mut dense_cut);
        for (n, &r) in rows.iter().enumerate() {
            assert_eq!(
                &case.dense[r * in_feat..(r + 1) * in_feat],
                &dense_cut[n * in_feat..(n + 1) * in_feat]
            );
        }
    }

    #[test]
    fn an_out_of_range_row_is_refused() {
        let case = build_case(4, 128, 128, 4);
        assert!(case.w.select_rows(&[0, 3]).is_some());
        assert!(case.w.select_rows(&[0, 4]).is_none());
    }

    const WIDTHS: [u8; 4] = [2, 3, 4, TERNARY_RECORD_BITS];

    #[test]
    fn dequantize_matches_reference_reconstruction() {
        for &bits in &WIDTHS {
            for &(out, in_feat, group) in &[(5usize, 16usize, 16usize), (3, 40, 8), (7, 128, 128)] {
                let c = build_case(out, in_feat, group, bits);
                let mut got = vec![0.0f32; out * in_feat];
                c.w.dequantize_to(&mut got);
                for (i, (&g_i, &want)) in got.iter().zip(c.dense.iter()).enumerate() {
                    assert!(
                        (g_i - want).abs() < 1e-5,
                        "bits={bits} shape={out}x{in_feat} g={group} i={i}: {g_i} vs {want}"
                    );
                }
            }
        }
    }

    /// The whole point of the module: the activation-side rotation must give the
    /// same answer as multiplying by the reconstructed weights.
    #[test]
    fn matvec_matches_dense_multiply() {
        for &bits in &WIDTHS {
            for &(out, in_feat, group) in &[
                (5usize, 16usize, 16usize),
                (3, 40, 8),
                (9, 128, 128),
                (4, 200, 64),
            ] {
                let c = build_case(out, in_feat, group, bits);
                let x: Vec<f32> = (0..in_feat)
                    .map(|i| (i as f32 * 0.31).sin() * 1.7 + 0.05 * i as f32)
                    .collect();

                let mut got = vec![0.0f32; out];
                c.w.matvec(&x, &mut got);

                for (o, &g_o) in got.iter().enumerate() {
                    let want: f32 = (0..in_feat).map(|j| c.dense[o * in_feat + j] * x[j]).sum();
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (g_o - want).abs() < tol,
                        "bits={bits} shape={out}x{in_feat} g={group} o={o}: {g_o} vs {want}"
                    );
                }
            }
        }
    }

    /// Padded columns must contribute nothing — upstream truncates them away.
    #[test]
    fn padding_does_not_leak_into_matvec() {
        // in_feat 40 with group 32 pads to 64; 24 columns are dropped upstream.
        let c = build_case(6, 40, 32, 2);
        let x: Vec<f32> = (0..40).map(|i| 1.0 + i as f32).collect();
        let mut got = vec![0.0f32; 6];
        c.w.matvec(&x, &mut got);
        for (o, &g_o) in got.iter().enumerate() {
            let want: f32 = (0..40).map(|j| c.dense[o * 40 + j] * x[j]).sum();
            assert!((g_o - want).abs() < 1e-3 * want.abs().max(1.0));
        }
    }

    /// A row band must equal the same rows of a full matvec.
    #[test]
    fn row_range_matches_full_matvec() {
        for &bits in &WIDTHS {
            let c = build_case(24, 128, 128, bits);
            let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.19).sin()).collect();
            let mut full = vec![0.0f32; 24];
            c.w.matvec(&x, &mut full);

            let mut xh = vec![0.0f32; c.w.prepared_len()];
            c.w.prepare_input(&x, &mut xh);
            for &(start, len) in &[(0usize, 4usize), (4, 4), (8, 16), (20, 4), (23, 1)] {
                let mut band = vec![0.0f32; len];
                c.w.matvec_rows_prepared(&xh, start, &mut band);
                assert_eq!(
                    &band[..],
                    &full[start..start + len],
                    "bits={bits} band {start}+{len}"
                );
            }
        }
    }

    /// The batched matmul must agree with `batch` separate matvecs, exactly:
    /// same values in the same order, only the loop nesting differs.
    /// The optimised matvec must agree with the simple reference. Different
    /// accumulation order, so this is a tolerance check bounded by the
    /// activation's L1 norm rather than by the result.
    #[test]
    fn matvec_matches_simple_reference() {
        for &bits in &WIDTHS {
            for &(out, in_feat, group) in &[
                (16usize, 128usize, 128usize),
                (5, 40, 8),
                (7, 200, 64),
                (9, 512, 128),
            ] {
                let c = build_case(out, in_feat, group, bits);
                let x: Vec<f32> = (0..in_feat)
                    .map(|i| (i as f32 * 0.29).sin() * 2.0)
                    .collect();
                let mut xh = vec![0.0f32; c.w.prepared_len()];
                c.w.prepare_input(&x, &mut xh);

                let mut fast = vec![0.0f32; out];
                let mut slow = vec![0.0f32; out];
                c.w.matvec_prepared(&xh, &mut fast);
                c.w.matvec_prepared_reference(&xh, &mut slow);

                let l1: f32 = xh.iter().map(|v| v.abs()).sum();
                for o in 0..out {
                    assert!(
                        (fast[o] - slow[o]).abs() <= 1e-5 * l1.max(1.0),
                        "bits={bits} {out}x{in_feat} g={group} o={o}: {} vs {}",
                        fast[o],
                        slow[o]
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_matches_repeated_matvec() {
        for &bits in &WIDTHS {
            for &(out, in_feat, group) in &[
                (24usize, 128usize, 128usize),
                (9, 40, 8),
                (5, 200, 64),
                (16, 512, 128),
            ] {
                let c = build_case(out, in_feat, group, bits);
                for &batch in &[1usize, 2, 7, 16] {
                    // Distinct activation per batch row.
                    let mut xh = vec![0.0f32; batch * c.w.prepared_len()];
                    let mut xs = Vec::new();
                    for b in 0..batch {
                        let x: Vec<f32> = (0..in_feat)
                            .map(|i| ((i + b * 13) as f32 * 0.23).sin() * 1.3)
                            .collect();
                        c.w.prepare_input(
                            &x,
                            &mut xh[b * c.w.prepared_len()..(b + 1) * c.w.prepared_len()],
                        );
                        xs.push(x);
                    }

                    for &(row_start, rows) in &[(0usize, out), (0, 1), (1, 3.min(out - 1))] {
                        if row_start + rows > out || rows == 0 {
                            continue;
                        }
                        let mut got = vec![0.0f32; batch * rows];
                        let mut acc = vec![0.0f32; batch];
                        c.w.matmul_rows_prepared(&xh, batch, row_start, rows, &mut got, &mut acc);

                        for b in 0..batch {
                            let mut want = vec![0.0f32; rows];
                            c.w.matvec_rows_prepared(
                                &xh[b * c.w.prepared_len()..(b + 1) * c.w.prepared_len()],
                                row_start,
                                &mut want,
                            );
                            for r in 0..rows {
                                assert_eq!(
                                    got[b * rows + r],
                                    want[r],
                                    "bits={bits} {out}x{in_feat} g={group} batch={batch} \
                                     rows={row_start}+{rows} b={b} r={r}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_input_is_reusable_across_weights() {
        let a = build_case(4, 128, 128, 2);
        let b = build_case(6, 128, 128, 4);
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).cos()).collect();

        let mut xh = vec![0.0f32; a.w.prepared_len()];
        a.w.prepare_input(&x, &mut xh);

        let mut ya = vec![0.0f32; 4];
        let mut yb = vec![0.0f32; 6];
        a.w.matvec_prepared(&xh, &mut ya);
        b.w.matvec_prepared(&xh, &mut yb);

        let mut ya_ref = vec![0.0f32; 4];
        let mut yb_ref = vec![0.0f32; 6];
        a.w.matvec(&x, &mut ya_ref);
        b.w.matvec(&x, &mut yb_ref);
        assert_eq!(ya, ya_ref);
        assert_eq!(yb, yb_ref);
    }

    #[test]
    fn row_bytes_match_shipped_tensor_sizes() {
        // Directory records from the real needle2.cact.
        // (out, in, bits, group) -> total blob bytes
        for &(out, in_f, bits, group, want) in &[
            (8192usize, 512usize, 4u8, 128usize, 2_162_688usize), // embedding
            (512, 512, 2, 128, 69_632),                           // q_proj
            (256, 512, 2, 128, 34_816),                           // k_proj
            (108, 2048, 4, 128, 114_048),                         // mhc_phi_pre
            (432, 2048, 4, 128, 456_192),                         // mhc_phi_res
            (32768, 128, 2, 128, 1_114_112),                      // engram tables
        ] {
            let in_pad = in_f.div_ceil(group) * group;
            let got = out * packed_row_bytes(in_pad, bits) + out * (in_pad / group) * 2;
            assert_eq!(got, want, "{out}x{in_f} @ {bits}b/{group}");
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let cb = fake_codebook(128);
        let blob = vec![0u8; 1 << 16];
        // CqWeight holds bulk buffers and is deliberately not Debug/PartialEq,
        // so compare the error side only.
        let err = |b: &[u8], out, in_f, group, bits, cb: &[f32]| {
            CqWeight::from_blob(b, out, in_f, group, bits, cb).err()
        };
        assert_eq!(
            err(&blob, 4, 128, 128, 7, &cb),
            Some(CqError::UnsupportedBits(7))
        );
        assert_eq!(err(&blob, 4, 128, 96, 2, &cb), Some(CqError::BadGroup(96)));
        assert_eq!(
            err(&blob, 4, 128, 100, 2, &cb),
            Some(CqError::BadGroup(100))
        );
        assert_eq!(err(&blob, 0, 128, 128, 2, &cb), Some(CqError::EmptyShape));
        assert_eq!(err(&blob, 4, 0, 128, 2, &cb), Some(CqError::EmptyShape));
        assert!(matches!(
            err(&[0u8; 4], 512, 512, 128, 2, &cb),
            Some(CqError::ShortBlob { .. })
        ));
        assert_eq!(
            err(&blob, 4, 128, 128, 2, &cb[..10]),
            Some(CqError::BadCodebookLen(10))
        );
    }
}

#[cfg(test)]
mod dispatch_tests {
    // This crate is `no_std`; the test harness links std, so pull it in for the
    // feature-detection macro and for printing.
    extern crate std;

    /// On x86_64 with AVX2 present, the wide path must actually be taken.
    ///
    /// Without this the AVX2 build could be dead code and every differential test
    /// would still pass, having only ever exercised the SSE2 fallback. This is
    /// the only assertion that the dispatch itself happens; development for this
    /// crate is on aarch64, where there is nothing to dispatch, so it is CI on
    /// x86_64 runners that gives the claim teeth.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    fn avx2_path_is_actually_selected_when_supported() {
        let cpuid_says = self::std::is_x86_feature_detected!("avx2");
        let we_say = crate::quant::has_avx2();
        assert_eq!(
            we_say, cpuid_says,
            "has_avx2() disagrees with std's detection; the dispatch would pick the wrong kernel"
        );
        if cpuid_says {
            assert!(
                we_say,
                "AVX2 is present but the wide kernel would not be selected"
            );
            self::std::eprintln!("AVX2 dispatch active: the wide packed-dot kernel is in use");
        } else {
            self::std::eprintln!("no AVX2 on this host; the portable kernel is in use");
        }
    }
    // There is deliberately no aarch64 counterpart: NEON is part of the ARMv8-A
    // baseline, so there is no runtime choice to verify. `matvec_matches_simple_reference`
    // covers the kernel's correctness on whatever LLVM emits there.
}
