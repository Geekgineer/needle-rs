//! Reader for the Needle v2 `.cact` weight container.
//!
//! Byte layout is specified by `needle/model/export.py`; this is a
//! reimplementation of `read_export()` plus the positional tensor canon its
//! docstring defines. Everything is little-endian.
//!
//! ```text
//! header      120 bytes: 29 u32 geometry fields then rope_theta as f32
//! codebook    codebook_len * f32   (cb2 | cb3 | cb4, already /sqrt(group))
//! directory   num_tensors * 44-byte records, NO names — tensors are positional
//! blobs       64-byte aligned
//! ```
//!
//! The directory carries no names, so a reader has to know the canonical order.
//! [`CactLayout`] derives every index from the header geometry, and
//! [`Cact::layout`] validates the shapes it lands on against that geometry, so
//! a layout drift surfaces as an error at load rather than as garbage logits.

use needle_core::cq::{self, CqWeight};
use needle_core::math::f16_to_f32;
use std::fmt;
use std::path::Path;

/// Magic in the first four bytes.
pub const TAG: u32 = 0x05E1_2A83;
/// Fixed header size: 29 u32 fields plus one f32.
pub const HEADER_BYTES: usize = 30 * 4;

/// Needle 3 container tag — one greater than [`TAG`], so a container states
/// its own generation in the first word and dispatch never guesses.
pub const TAG_V3: u32 = 0x05E1_2A84;

/// Needle 3 header: 48 `u32` then `rope_theta` as `f32`
/// (`_HDR_FMT = "<48If"` in upstream `export.py`).
pub const HEADER_BYTES_V3: usize = 49 * 4;
/// Directory record size.
pub const REC_BYTES: usize = 44;
/// Blob alignment.
pub const ALIGN: usize = 64;

/// Directory `dtype` codes.
pub const DT_FP16: u8 = 1;
pub const DT_FP32: u8 = 2;
pub const DT_CQ: u8 = 3;
pub const DT_RAW: u8 = 4;

/// Head codes in `heads.manifest`, in canonical order.
pub const HEAD_CONTRASTIVE: u8 = 1;
pub const HEAD_CONFIDENCE: u8 = 2;

/// Per-layer tensor count in the canon: `norm_in, q_proj, k_proj, v_proj,
/// q_norm, k_norm, gate_proj, out_proj, post_norm, attn_gate, pre_hada,
/// d1, d2, d3`.
pub const TENSORS_PER_LAYER: usize = 14;
/// The nine mHC blocks that follow the layers.
pub const MHC_TENSORS: usize = 9;
/// Per engram site: `tables, key_proj, value_proj, taps`.
pub const TENSORS_PER_ENGRAM_SITE: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub enum CactError {
    TooShort {
        need: usize,
        got: usize,
    },
    BadTag(u32),
    BadCodebookLen(usize),
    /// A directory record points outside the file.
    BlobOutOfRange {
        index: usize,
        offset: u64,
        nbytes: u64,
        file: usize,
    },
    BadNdim {
        index: usize,
        ndim: u8,
    },
    /// Tensor `index` has dtype `got` where the caller needed `want`.
    DtypeMismatch {
        index: usize,
        want: u8,
        got: u8,
    },
    /// Byte length is not a whole number of elements for the dtype.
    RaggedBlob {
        index: usize,
        nbytes: u64,
    },
    /// The canon needs more tensors than the directory holds.
    TensorCountTooSmall {
        need: usize,
        got: usize,
    },
    /// A canon slot has a shape the header geometry does not predict.
    ShapeMismatch {
        index: usize,
        what: &'static str,
        want: [usize; 4],
        got: [usize; 4],
    },
    /// Geometry field that cannot be zero, is.
    BadGeometry(&'static str),
    /// `heads.manifest` length is not consistent with the trailing tensor count.
    BadHeadManifest {
        extra: usize,
    },
    /// Unrecognised head code in `heads.manifest`.
    UnknownHeadCode(u8),
    Cq(cq::CqError),
}

impl From<cq::CqError> for CactError {
    fn from(e: cq::CqError) -> Self {
        CactError::Cq(e)
    }
}

impl fmt::Display for CactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { need, got } => {
                write!(f, "cact truncated: need at least {need} bytes, got {got}")
            }
            Self::BadTag(t) => write!(f, "not a cact blob: tag {t:#010x}, want {TAG:#010x}"),
            Self::BadCodebookLen(n) => write!(
                f,
                "header codebook is {n} floats, want {} (cb2|cb3|cb4)",
                cq::CODEBOOK_LEN
            ),
            Self::BlobOutOfRange {
                index,
                offset,
                nbytes,
                file,
            } => write!(
                f,
                "tensor {index} spans {offset}..{} but the file is {file} bytes",
                offset + nbytes
            ),
            Self::BadNdim { index, ndim } => {
                write!(f, "tensor {index} has ndim {ndim}, want 0..=4")
            }
            Self::DtypeMismatch { index, want, got } => {
                write!(f, "tensor {index} has dtype {got}, want {want}")
            }
            Self::RaggedBlob { index, nbytes } => {
                write!(
                    f,
                    "tensor {index} has {nbytes} bytes, not a whole element count"
                )
            }
            Self::TensorCountTooSmall { need, got } => write!(
                f,
                "the canon for this geometry needs {need} tensors, directory has {got}"
            ),
            Self::ShapeMismatch {
                index,
                what,
                want,
                got,
            } => write!(
                f,
                "tensor {index} ({what}): shape {got:?} does not match geometry {want:?}"
            ),
            Self::BadGeometry(field) => write!(f, "header field {field} must be non-zero"),
            Self::BadHeadManifest { extra } => write!(
                f,
                "{extra} trailing tensors is not 1 + 3*heads for any head count"
            ),
            Self::UnknownHeadCode(c) => write!(f, "unknown probe-head code {c}"),
            Self::Cq(e) => write!(f, "cq: {e:?}"),
        }
    }
}

impl std::error::Error for CactError {}

/// Model geometry, straight out of the header. One binary runs any
/// configuration of the architecture, so nothing here is a compile-time
/// constant.
#[derive(Debug, Clone, PartialEq)]
pub struct CactGeometry {
    pub num_tensors: usize,
    pub codebook_len: usize,
    /// Sliding-window width the model was trained with. `0` means size it from
    /// the KV budget alone.
    pub kv_window: usize,
    /// KV-cache width the model was post-trained for. `8` means int8.
    pub kv_bits: u32,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Hadamard width inside HadamardMLP: `d_model` rounded up to a power of two.
    pub hada_n: usize,
    pub mhc_lanes: usize,
    pub engram_slots: usize,
    pub engram_sub_dim: usize,
    pub num_engram_tables: usize,
    pub engram_conv_taps: usize,
    pub engram_conv_dilation: usize,
    pub engram_orders: Vec<usize>,
    /// Layer indices carrying an engram site.
    pub engram_sites: Vec<usize>,
    pub rope_theta: f32,
}

impl CactGeometry {
    /// Attention width — `num_heads * head_dim`.
    pub fn attn_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Width of the packed K or V projection.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Q heads served by each KV head.
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Flattened mHC lane-stream width, `mhc_lanes * d_model`.
    pub fn mhc_width(&self) -> usize {
        self.mhc_lanes * self.d_model
    }

    /// Token history the engram convolution reaches back over:
    /// `conv_taps * max(orders)`. Mirrors `decode._engram_window`.
    pub fn engram_window(&self) -> usize {
        if self.engram_sites.is_empty() {
            0
        } else {
            self.engram_conv_taps * self.engram_orders.iter().copied().max().unwrap_or(0)
        }
    }

    fn validate(&self) -> Result<(), CactError> {
        for (v, name) in [
            (self.vocab_size, "vocab"),
            (self.d_model, "d_model"),
            (self.num_heads, "num_heads"),
            (self.num_kv_heads, "num_kv_heads"),
            (self.num_layers, "num_layers"),
            (self.head_dim, "head_dim"),
            (self.max_seq_len, "max_seq_len"),
            (self.mhc_lanes, "mhc_lanes"),
        ] {
            if v == 0 {
                return Err(CactError::BadGeometry(name));
            }
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(CactError::BadGeometry("num_heads % num_kv_heads"));
        }
        if self.hada_n < self.d_model || !self.hada_n.is_power_of_two() {
            return Err(CactError::BadGeometry("hada_n"));
        }
        if !self.engram_sites.is_empty() {
            if self.engram_slots == 0 || self.engram_sub_dim == 0 || self.num_engram_tables == 0 {
                return Err(CactError::BadGeometry("engram geometry"));
            }
            if self.engram_sites.iter().any(|&l| l >= self.num_layers) {
                return Err(CactError::BadGeometry("engram_sites beyond num_layers"));
            }
        }
        Ok(())
    }
}

/// One directory record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub dtype: u8,
    pub ndim: u8,
    pub shape: [usize; 4],
    pub offset: u64,
    pub nbytes: u64,
    pub group: usize,
    /// Record width for CQ tensors: 2, 3, 4, or 5 for ternary.
    pub bits: u8,
}

impl Record {
    /// Element count implied by `shape[..ndim]`.
    pub fn numel(&self) -> usize {
        self.shape[..self.ndim as usize]
            .iter()
            .product::<usize>()
            .max(1)
    }
}

/// A parsed `.cact` blob. Owns the bytes; tensors are decoded on demand.
/// Geometry declared by a Needle 3 container header.
///
/// Field order is fixed by upstream `export.py`; the runtime derives
/// everything from it, so one binary runs any configuration of the
/// architecture. Read from the container, never assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct CactV3Geometry {
    pub num_tensors: usize,
    pub codebook_len: usize,
    pub kv_window: usize,
    pub kv_bits: u32,
    pub vocab_size: usize,
    /// Tied text-slice head; 0 means the full vocab. Rows past this are
    /// input-only code embeddings.
    pub out_vocab: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    /// Query/key head width. Unlike v2, this differs from `v_head_dim`.
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub hada_n: usize,
    pub mhc_lanes: usize,
    /// Per-layer local attention width. Layers in `global_layers` ignore it.
    pub sliding_window: usize,
    /// Bit `i` marks layer `i` as full-attention.
    global_mask: u64,
    /// Causal depthwise convolution width over Q, K and V. 0 means none.
    pub qkv_conv_taps: usize,
    pub engram_slots: usize,
    pub engram_sub_dim: usize,
    pub num_engram_tables: usize,
    pub engram_conv_taps: usize,
    pub engram_conv_dilation: usize,
    pub engram_seed_heads: usize,
    pub engram_orders: Vec<usize>,
    pub engram_sites: Vec<usize>,
    pub rope_theta: f32,
}

impl CactV3Geometry {
    /// Parse the 49 header words.
    pub fn from_words(w: &[u32]) -> Result<Self, CactError> {
        if w.len() < 49 {
            return Err(CactError::TooShort {
                need: HEADER_BYTES_V3,
                got: w.len() * 4,
            });
        }
        if w[0] != TAG_V3 {
            return Err(CactError::BadTag(w[0]));
        }
        let num_orders = w[26] as usize;
        let num_sites = w[31] as usize;
        let take = |base: usize, n: usize, cap: usize| -> Vec<usize> {
            (0..n.min(cap)).map(|k| w[base + k] as usize).collect()
        };
        Ok(Self {
            num_tensors: w[1] as usize,
            codebook_len: w[2] as usize,
            kv_window: w[3] as usize,
            kv_bits: w[4],
            vocab_size: w[5] as usize,
            out_vocab: w[6] as usize,
            d_model: w[7] as usize,
            num_heads: w[8] as usize,
            num_kv_heads: w[9] as usize,
            num_layers: w[10] as usize,
            qk_head_dim: w[11] as usize,
            v_head_dim: w[12] as usize,
            max_seq_len: w[13] as usize,
            hada_n: w[14] as usize,
            mhc_lanes: w[15] as usize,
            sliding_window: w[16] as usize,
            global_mask: (w[17] as u64) | ((w[18] as u64) << 32),
            qkv_conv_taps: w[19] as usize,
            engram_slots: w[20] as usize,
            engram_sub_dim: w[21] as usize,
            num_engram_tables: w[22] as usize,
            engram_conv_taps: w[23] as usize,
            engram_conv_dilation: w[24] as usize,
            engram_seed_heads: w[25] as usize,
            engram_orders: take(27, num_orders, 4),
            engram_sites: take(32, num_sites, 16),
            rope_theta: f32::from_bits(w[48]),
        })
    }

    /// Layers that attend over the whole sequence rather than
    /// `sliding_window`.
    pub fn global_layers(&self) -> Vec<usize> {
        (0..self.num_layers)
            .filter(|i| self.is_global(*i))
            .collect()
    }

    /// True when layer `i` attends globally.
    pub fn is_global(&self, i: usize) -> bool {
        i < 64 && self.global_mask >> i & 1 == 1
    }

    /// Queries per key/value head.
    pub fn kv_repeat(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Width of the concatenated query projection.
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.qk_head_dim
    }

    /// Width of the concatenated key projection.
    pub fn k_dim(&self) -> usize {
        self.num_kv_heads * self.qk_head_dim
    }

    /// Width of the concatenated value projection.
    pub fn v_dim(&self) -> usize {
        self.num_kv_heads * self.v_head_dim
    }
}

/// Parse `num_tensors` directory records starting at `dir_start`.
///
/// The record layout is identical across container generations — only the
/// header ahead of it changed — so both loaders use this.
fn parse_directory(
    raw: &[u8],
    dir_start: usize,
    num_tensors: usize,
) -> Result<Vec<Record>, CactError> {
    let mut records = Vec::with_capacity(num_tensors);
    for i in 0..num_tensors {
        let b = &raw[dir_start + i * REC_BYTES..dir_start + (i + 1) * REC_BYTES];
        let g32 = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let g64 = |o: usize| {
            u64::from_le_bytes([
                b[o],
                b[o + 1],
                b[o + 2],
                b[o + 3],
                b[o + 4],
                b[o + 5],
                b[o + 6],
                b[o + 7],
            ])
        };
        let ndim = b[1];
        if ndim > 4 {
            return Err(CactError::BadNdim { index: i, ndim });
        }
        // Layout: u8 dtype, u8 ndim, u16 pad, u32 shape[4], u64 offset,
        //         u64 nbytes, u32 group, u32 bits.
        let rec = Record {
            dtype: b[0],
            ndim,
            shape: [
                g32(4) as usize,
                g32(8) as usize,
                g32(12) as usize,
                g32(16) as usize,
            ],
            offset: g64(20),
            nbytes: g64(28),
            group: g32(36) as usize,
            bits: g32(40) as u8,
        };
        let end = rec.offset.saturating_add(rec.nbytes);
        if end > raw.len() as u64 {
            return Err(CactError::BlobOutOfRange {
                index: i,
                offset: rec.offset,
                nbytes: rec.nbytes,
                file: raw.len(),
            });
        }
        records.push(rec);
    }
    Ok(records)
}

/// Decode the shared Lloyd-Max codebooks (`cb2|cb3|cb4`) that sit between
/// the header and the directory.
fn parse_codebook(raw: &[u8], header_bytes: usize, codebook_len: usize) -> Vec<f32> {
    raw[header_bytes..header_bytes + codebook_len * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub struct Cact {
    raw: Vec<u8>,
    pub geom: CactGeometry,
    /// Concatenated `cb2|cb3|cb4`, exactly as stored.
    pub codebook: Vec<f32>,
    records: Vec<Record>,
}

impl Cact {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let raw = std::fs::read(path)?;
        Self::from_bytes(raw).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_bytes(raw: Vec<u8>) -> Result<Self, CactError> {
        if raw.len() < HEADER_BYTES {
            return Err(CactError::TooShort {
                need: HEADER_BYTES,
                got: raw.len(),
            });
        }
        let u = |i: usize| -> u32 {
            u32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
        };
        if u(0) != TAG {
            return Err(CactError::BadTag(u(0)));
        }
        let num_tensors = u(1) as usize;
        let codebook_len = u(2) as usize;
        let num_orders = u(19) as usize;
        let num_sites = u(24) as usize;
        // orders[4] at 20..24, sites[4] at 25..29; counts index into them.
        let take = |base: usize, n: usize| -> Vec<usize> {
            (0..n.min(4)).map(|k| u(base + k) as usize).collect()
        };
        let geom = CactGeometry {
            num_tensors,
            codebook_len,
            kv_window: u(3) as usize,
            kv_bits: u(4),
            vocab_size: u(5) as usize,
            d_model: u(6) as usize,
            num_heads: u(7) as usize,
            num_kv_heads: u(8) as usize,
            num_layers: u(9) as usize,
            head_dim: u(10) as usize,
            max_seq_len: u(11) as usize,
            hada_n: u(12) as usize,
            mhc_lanes: u(13) as usize,
            engram_slots: u(14) as usize,
            engram_sub_dim: u(15) as usize,
            num_engram_tables: u(16) as usize,
            engram_conv_taps: u(17) as usize,
            engram_conv_dilation: u(18) as usize,
            engram_orders: take(20, num_orders),
            engram_sites: take(25, num_sites),
            rope_theta: f32::from_bits(u(29)),
        };
        geom.validate()?;

        if codebook_len != cq::CODEBOOK_LEN {
            return Err(CactError::BadCodebookLen(codebook_len));
        }
        let dir_start = HEADER_BYTES + codebook_len * 4;
        let dir_end = dir_start + num_tensors * REC_BYTES;
        if raw.len() < dir_end {
            return Err(CactError::TooShort {
                need: dir_end,
                got: raw.len(),
            });
        }

        let codebook = parse_codebook(&raw, HEADER_BYTES, codebook_len);

        let records = parse_directory(&raw, dir_start, num_tensors)?;

        Ok(Self {
            raw,
            geom,
            codebook,
            records,
        })
    }

    pub fn num_tensors(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn record(&self, i: usize) -> &Record {
        &self.records[i]
    }

    /// Total file size, for size accounting.
    pub fn byte_len(&self) -> usize {
        self.raw.len()
    }

    fn blob(&self, i: usize) -> &[u8] {
        let r = &self.records[i];
        &self.raw[r.offset as usize..(r.offset + r.nbytes) as usize]
    }

    /// Decode an FP16 or FP32 tensor to `f32`.
    pub fn floats(&self, i: usize) -> Result<Vec<f32>, CactError> {
        let r = &self.records[i];
        let blob = self.blob(i);
        match r.dtype {
            DT_FP16 => {
                if !r.nbytes.is_multiple_of(2) {
                    return Err(CactError::RaggedBlob {
                        index: i,
                        nbytes: r.nbytes,
                    });
                }
                Ok(blob
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            DT_FP32 => {
                if !r.nbytes.is_multiple_of(4) {
                    return Err(CactError::RaggedBlob {
                        index: i,
                        nbytes: r.nbytes,
                    });
                }
                Ok(blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            got => Err(CactError::DtypeMismatch {
                index: i,
                want: DT_FP16,
                got,
            }),
        }
    }

    /// Parse a CQ tensor. Shape must be 2-D `[out, in]`.
    pub fn cq(&self, i: usize) -> Result<CqWeight, CactError> {
        let r = &self.records[i];
        if r.dtype != DT_CQ {
            return Err(CactError::DtypeMismatch {
                index: i,
                want: DT_CQ,
                got: r.dtype,
            });
        }
        Ok(CqWeight::from_blob(
            self.blob(i),
            r.shape[0],
            r.shape[1],
            r.group,
            r.bits,
            &self.codebook,
        )?)
    }

    /// Borrow a RAW tensor's bytes (the embedded tokenizer).
    pub fn raw_tensor(&self, i: usize) -> Result<&[u8], CactError> {
        let r = &self.records[i];
        if r.dtype != DT_RAW {
            return Err(CactError::DtypeMismatch {
                index: i,
                want: DT_RAW,
                got: r.dtype,
            });
        }
        Ok(self.blob(i))
    }

    /// Resolve the positional tensor canon and check it against the geometry.
    pub fn layout(&self) -> Result<CactLayout, CactError> {
        CactLayout::derive(self)
    }
}

/// Indices of one transformer layer's tensors.
#[derive(Debug, Clone, Copy)]
pub struct LayerIdx {
    pub norm_in: usize,
    pub q_proj: usize,
    pub k_proj: usize,
    pub v_proj: usize,
    pub q_norm: usize,
    pub k_norm: usize,
    pub gate_proj: usize,
    pub out_proj: usize,
    pub post_norm: usize,
    pub attn_gate: usize,
    pub pre_hada: usize,
    pub d1: usize,
    pub d2: usize,
    pub d3: usize,
}

/// Indices of one engram site's tensors.
#[derive(Debug, Clone, Copy)]
pub struct EngramIdx {
    pub tables: usize,
    pub key_proj: usize,
    pub value_proj: usize,
    pub taps: usize,
}

/// Indices of the nine mHC blocks.
#[derive(Debug, Clone, Copy)]
pub struct MhcIdx {
    pub a_pre: usize,
    pub a_post: usize,
    pub a_res: usize,
    pub b_pre: usize,
    pub b_post: usize,
    pub b_res: usize,
    pub phi_pre: usize,
    pub phi_post: usize,
    pub phi_res: usize,
}

/// Indices of one probe head's tensors.
#[derive(Debug, Clone, Copy)]
pub struct HeadIdx {
    pub code: u8,
    pub probes: usize,
    pub proj: usize,
    pub bias: usize,
}

/// Every tensor index the canon defines, resolved for one blob.
#[derive(Debug, Clone)]
pub struct CactLayout {
    pub embedding: usize,
    pub layers: Vec<LayerIdx>,
    pub mhc: MhcIdx,
    pub engrams: Vec<EngramIdx>,
    pub final_norm: usize,
    /// `heads.manifest`, present only when probe heads were exported.
    pub head_manifest: Option<usize>,
    pub heads: Vec<HeadIdx>,
    /// The embedded tokenizer, when present.
    pub tokenizer: Option<usize>,
}

impl CactLayout {
    fn derive(c: &Cact) -> Result<Self, CactError> {
        let g = &c.geom;
        let n_sites = g.engram_sites.len();
        let through_final_norm = 1
            + g.num_layers * TENSORS_PER_LAYER
            + MHC_TENSORS
            + n_sites * TENSORS_PER_ENGRAM_SITE
            + 1;
        if c.num_tensors() < through_final_norm {
            return Err(CactError::TensorCountTooSmall {
                need: through_final_norm,
                got: c.num_tensors(),
            });
        }

        // The tokenizer, when exported, is the last tensor and the only RAW one.
        let tokenizer = c
            .records
            .last()
            .filter(|r| r.dtype == DT_RAW)
            .map(|_| c.num_tensors() - 1);

        let extra = c.num_tensors() - through_final_norm - usize::from(tokenizer.is_some());
        // extra == 0 -> no heads. Otherwise: manifest + 3 tensors per head.
        let n_heads = if extra == 0 {
            0
        } else if extra >= 4 && (extra - 1).is_multiple_of(3) {
            (extra - 1) / 3
        } else {
            return Err(CactError::BadHeadManifest { extra });
        };

        let mut i = 0usize;
        let embedding = i;
        i += 1;

        let mut layers = Vec::with_capacity(g.num_layers);
        for _ in 0..g.num_layers {
            layers.push(LayerIdx {
                norm_in: i,
                q_proj: i + 1,
                k_proj: i + 2,
                v_proj: i + 3,
                q_norm: i + 4,
                k_norm: i + 5,
                gate_proj: i + 6,
                out_proj: i + 7,
                post_norm: i + 8,
                attn_gate: i + 9,
                pre_hada: i + 10,
                d1: i + 11,
                d2: i + 12,
                d3: i + 13,
            });
            i += TENSORS_PER_LAYER;
        }

        // export._tensors writes the a_*/b_* scalars first, then the phi matrices.
        let mhc = MhcIdx {
            a_pre: i,
            a_post: i + 1,
            a_res: i + 2,
            b_pre: i + 3,
            b_post: i + 4,
            b_res: i + 5,
            phi_pre: i + 6,
            phi_post: i + 7,
            phi_res: i + 8,
        };
        i += MHC_TENSORS;

        let mut engrams = Vec::with_capacity(n_sites);
        for _ in 0..n_sites {
            engrams.push(EngramIdx {
                tables: i,
                key_proj: i + 1,
                value_proj: i + 2,
                taps: i + 3,
            });
            i += TENSORS_PER_ENGRAM_SITE;
        }

        let final_norm = i;
        i += 1;

        let (head_manifest, heads) = if n_heads == 0 {
            (None, Vec::new())
        } else {
            let manifest_idx = i;
            let codes = c.floats(manifest_idx)?;
            if codes.len() != n_heads {
                return Err(CactError::BadHeadManifest { extra });
            }
            i += 1;
            let mut heads = Vec::with_capacity(n_heads);
            for &code in &codes {
                let code = code as u8;
                if code != HEAD_CONTRASTIVE && code != HEAD_CONFIDENCE {
                    return Err(CactError::UnknownHeadCode(code));
                }
                heads.push(HeadIdx {
                    code,
                    probes: i,
                    proj: i + 1,
                    bias: i + 2,
                });
                i += 3;
            }
            (Some(manifest_idx), heads)
        };

        let layout = Self {
            embedding,
            layers,
            mhc,
            engrams,
            final_norm,
            head_manifest,
            heads,
            tokenizer,
        };
        layout.check_shapes(c)?;
        Ok(layout)
    }

    /// Assert every canon slot has the shape the header geometry predicts.
    ///
    /// The directory is nameless, so this is the only thing standing between a
    /// canon drift and silently reading `k_proj` as `v_proj`.
    fn check_shapes(&self, c: &Cact) -> Result<(), CactError> {
        let g = &c.geom;
        let d = g.d_model;

        let want = |i: usize, what: &'static str, dims: &[usize]| -> Result<(), CactError> {
            let r = c.record(i);
            let got_ndim = r.ndim as usize;
            let mut want4 = [0usize; 4];
            want4[..dims.len()].copy_from_slice(dims);
            let mut got4 = [0usize; 4];
            got4[..got_ndim.min(4)].copy_from_slice(&r.shape[..got_ndim.min(4)]);
            if got_ndim != dims.len() || got4 != want4 {
                return Err(CactError::ShapeMismatch {
                    index: i,
                    what,
                    want: want4,
                    got: got4,
                });
            }
            Ok(())
        };

        want(self.embedding, "embedding", &[g.vocab_size, d])?;

        for l in &self.layers {
            want(l.norm_in, "norm_in", &[d])?;
            want(l.q_proj, "q_proj", &[g.attn_dim(), d])?;
            want(l.k_proj, "k_proj", &[g.kv_dim(), d])?;
            want(l.v_proj, "v_proj", &[g.kv_dim(), d])?;
            want(l.q_norm, "q_norm", &[g.head_dim])?;
            want(l.k_norm, "k_norm", &[g.head_dim])?;
            want(l.gate_proj, "gate_proj", &[g.attn_dim(), d])?;
            want(l.out_proj, "out_proj", &[d, g.attn_dim()])?;
            want(l.post_norm, "post_norm", &[d])?;
            want(l.attn_gate, "attn_gate", &[1])?;
            want(l.pre_hada, "pre_hada", &[d])?;
            want(l.d1, "d1", &[g.hada_n])?;
            want(l.d2, "d2", &[g.hada_n])?;
            want(l.d3, "d3", &[g.hada_n])?;
        }

        let (l_, n) = (g.num_layers, g.mhc_lanes);
        want(self.mhc.a_pre, "mhc_a_pre", &[l_])?;
        want(self.mhc.a_post, "mhc_a_post", &[l_])?;
        want(self.mhc.a_res, "mhc_a_res", &[l_])?;
        want(self.mhc.b_pre, "mhc_b_pre", &[l_, n])?;
        want(self.mhc.b_post, "mhc_b_post", &[l_, n])?;
        want(self.mhc.b_res, "mhc_b_res", &[l_, n, n])?;
        // phi are stored transposed and flattened: [L*lanes, lanes*d_model].
        want(self.mhc.phi_pre, "mhc_phi_pre", &[l_ * n, g.mhc_width()])?;
        want(self.mhc.phi_post, "mhc_phi_post", &[l_ * n, g.mhc_width()])?;
        want(
            self.mhc.phi_res,
            "mhc_phi_res",
            &[l_ * n * n, g.mhc_width()],
        )?;

        for e in &self.engrams {
            want(
                e.tables,
                "engram tables",
                &[g.num_engram_tables * g.engram_slots, g.engram_sub_dim],
            )?;
            let e_in = g.num_engram_tables * g.engram_sub_dim;
            want(e.key_proj, "engram key_proj", &[d, e_in])?;
            want(e.value_proj, "engram value_proj", &[d, e_in])?;
            want(e.taps, "engram taps", &[g.engram_conv_taps, d])?;
        }

        want(self.final_norm, "final_norm", &[d])?;

        for h in &self.heads {
            let probes = match c.record(h.probes).shape[0] {
                0 => {
                    return Err(CactError::ShapeMismatch {
                        index: h.probes,
                        what: "head probes",
                        want: [1, d, 0, 0],
                        got: [0, 0, 0, 0],
                    })
                }
                p => p,
            };
            want(h.probes, "head probes", &[probes, d])?;
            let out = c.record(h.proj).shape[0];
            want(h.proj, "head proj", &[out, probes * d])?;
            want(h.bias, "head bias", &[out])?;
        }
        Ok(())
    }

    /// The engram site index for a layer, if that layer carries one.
    pub fn engram_site_of_layer(geom: &CactGeometry, layer: usize) -> Option<usize> {
        geom.engram_sites.iter().position(|&l| l == layer)
    }

    pub fn head(&self, code: u8) -> Option<&HeadIdx> {
        self.heads.iter().find(|h| h.code == code)
    }
}

/// A loaded Needle 3 container.
///
/// Same body as v2 — codebook, nameless positional directory, 64-byte
/// aligned blobs — over a 196-byte header instead of 120. The generation
/// is decided by the tag, so a v2 container cannot be opened here and a v3
/// container cannot be opened as v2.
pub struct CactV3 {
    raw: Vec<u8>,
    pub geom: CactV3Geometry,
    /// Concatenated `cb2|cb3|cb4`, exactly as stored.
    pub codebook: Vec<f32>,
    records: Vec<Record>,
}

impl CactV3 {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let raw = std::fs::read(path)?;
        Self::from_bytes(raw).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn from_bytes(raw: Vec<u8>) -> Result<Self, CactError> {
        if raw.len() < HEADER_BYTES_V3 {
            return Err(CactError::TooShort {
                need: HEADER_BYTES_V3,
                got: raw.len(),
            });
        }
        let words: Vec<u32> = (0..49)
            .map(|i| {
                u32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]])
            })
            .collect();
        let geom = CactV3Geometry::from_words(&words)?;

        if geom.codebook_len != cq::CODEBOOK_LEN {
            return Err(CactError::BadCodebookLen(geom.codebook_len));
        }
        let dir_start = HEADER_BYTES_V3 + geom.codebook_len * 4;
        let dir_end = dir_start + geom.num_tensors * REC_BYTES;
        if raw.len() < dir_end {
            return Err(CactError::TooShort {
                need: dir_end,
                got: raw.len(),
            });
        }

        let codebook = parse_codebook(&raw, HEADER_BYTES_V3, geom.codebook_len);
        let records = parse_directory(&raw, dir_start, geom.num_tensors)?;

        Ok(Self {
            raw,
            geom,
            codebook,
            records,
        })
    }

    pub fn num_tensors(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn record(&self, i: usize) -> &Record {
        &self.records[i]
    }

    /// Total file size, for size accounting.
    pub fn byte_len(&self) -> usize {
        self.raw.len()
    }

    fn blob(&self, i: usize) -> &[u8] {
        let r = &self.records[i];
        &self.raw[r.offset as usize..(r.offset + r.nbytes) as usize]
    }

    /// The single `RAW` record, which carries the embedded tokenizer.
    pub fn raw_tensor(&self, i: usize) -> Result<&[u8], CactError> {
        let r = &self.records[i];
        if r.dtype != DT_RAW {
            return Err(CactError::DtypeMismatch {
                index: i,
                want: DT_RAW,
                got: r.dtype,
            });
        }
        Ok(self.blob(i))
    }

    /// Index of the embedded tokenizer blob, if the container has one.
    pub fn tokenizer_index(&self) -> Option<usize> {
        self.records.iter().position(|r| r.dtype == DT_RAW)
    }

    /// The embedded tokenizer blob.
    pub fn tokenizer_blob(&self) -> Option<&[u8]> {
        self.tokenizer_index().map(|i| self.blob(i))
    }

    pub fn cq(&self, i: usize) -> Result<CqWeight, CactError> {
        let r = &self.records[i];
        if r.dtype != DT_CQ {
            return Err(CactError::DtypeMismatch {
                index: i,
                want: DT_CQ,
                got: r.dtype,
            });
        }
        Ok(CqWeight::from_blob(
            self.blob(i),
            r.shape[0],
            r.shape[1],
            r.group,
            r.bits,
            &self.codebook,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal well-formed blob so the reader can be tested without the
    /// 13.7 MB checkpoint. Only FP16 tensors and no engram sites, so the canon
    /// stays small; CQ decoding is covered by `needle_core::cq` tests and by
    /// `tests/cact_parity.rs` against the real file.
    fn synth(num_layers: usize, with_heads: bool, with_tokenizer: bool) -> Vec<u8> {
        let (d, heads, kv_heads, hd, lanes) = (8usize, 2usize, 1usize, 4usize, 2usize);
        let vocab = 6usize;
        let attn = heads * hd;
        let kv = kv_heads * hd;
        let hada = d.next_power_of_two();

        // (shape, dtype) in canon order; embedding is FP16 here for simplicity.
        let mut shapes: Vec<Vec<usize>> = vec![vec![vocab, d]];
        for _ in 0..num_layers {
            shapes.extend([
                vec![d],
                vec![attn, d],
                vec![kv, d],
                vec![kv, d],
                vec![hd],
                vec![hd],
                vec![attn, d],
                vec![d, attn],
                vec![d],
                vec![1],
                vec![d],
                vec![hada],
                vec![hada],
                vec![hada],
            ]);
        }
        shapes.extend([
            vec![num_layers],
            vec![num_layers],
            vec![num_layers],
            vec![num_layers, lanes],
            vec![num_layers, lanes],
            vec![num_layers, lanes, lanes],
            vec![num_layers * lanes, lanes * d],
            vec![num_layers * lanes, lanes * d],
            vec![num_layers * lanes * lanes, lanes * d],
        ]);
        shapes.push(vec![d]); // final_norm
        if with_heads {
            shapes.push(vec![2]); // manifest: contrastive + confidence
            for &p in &[4usize, 8usize] {
                shapes.push(vec![p, d]);
                shapes.push(vec![1, p * d]);
                shapes.push(vec![1]);
            }
        }
        let n_tensors = shapes.len() + usize::from(with_tokenizer);

        let cb: Vec<f32> = (0..cq::CODEBOOK_LEN).map(|i| i as f32 * 0.01).collect();
        let mut hdr: Vec<u8> = Vec::new();
        for v in [
            TAG,
            n_tensors as u32,
            cq::CODEBOOK_LEN as u32,
            0,
            8,
            vocab as u32,
            d as u32,
            heads as u32,
            kv_heads as u32,
            num_layers as u32,
            hd as u32,
            32,
            hada as u32,
            lanes as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ] {
            hdr.extend_from_slice(&v.to_le_bytes());
        }
        hdr.extend_from_slice(&100000.0f32.to_le_bytes());
        assert_eq!(hdr.len(), HEADER_BYTES);
        for &v in &cb {
            hdr.extend_from_slice(&v.to_le_bytes());
        }

        // Lay out blobs 64-byte aligned after the directory.
        let mut pos = hdr.len() + n_tensors * REC_BYTES;
        let mut recs = Vec::new();
        for (ti, s) in shapes.iter().enumerate() {
            let numel: usize = s.iter().product();
            let nbytes = numel * 2;
            pos = (pos + ALIGN - 1) & !(ALIGN - 1);
            // manifest holds the head codes; everything else is filler.
            let vals: Vec<u16> = if with_heads && ti == shapes.len() - 7 {
                vec![0x3C00, 0x4000] // 1.0, 2.0
            } else {
                (0..numel)
                    .map(|k| 0x3C00u16.wrapping_add(k as u16))
                    .collect()
            };
            recs.push((
                DT_FP16,
                s.clone(),
                pos as u64,
                nbytes as u64,
                0usize,
                0u8,
                vals,
            ));
            pos += nbytes;
        }
        if with_tokenizer {
            pos = (pos + ALIGN - 1) & !(ALIGN - 1);
            recs.push((DT_RAW, vec![], pos as u64, 16, 0, 0, vec![0u16; 8]));
        }

        let mut dir = Vec::new();
        for (dtype, shape, offset, nbytes, group, bits, _) in &recs {
            dir.push(*dtype);
            dir.push(shape.len() as u8);
            dir.extend_from_slice(&0u16.to_le_bytes());
            for k in 0..4 {
                dir.extend_from_slice(&(*shape.get(k).unwrap_or(&0) as u32).to_le_bytes());
            }
            dir.extend_from_slice(&offset.to_le_bytes());
            dir.extend_from_slice(&nbytes.to_le_bytes());
            dir.extend_from_slice(&(*group as u32).to_le_bytes());
            dir.extend_from_slice(&(*bits as u32).to_le_bytes());
        }

        let mut buf = hdr;
        buf.extend_from_slice(&dir);
        for (_, _, offset, _, _, _, vals) in &recs {
            buf.resize(*offset as usize, 0);
            for v in vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn parses_header_and_directory() {
        let c = Cact::from_bytes(synth(3, true, true)).unwrap();
        assert_eq!(c.geom.d_model, 8);
        assert_eq!(c.geom.num_layers, 3);
        assert_eq!(c.geom.mhc_lanes, 2);
        assert_eq!(c.geom.rope_theta, 100000.0);
        assert_eq!(c.geom.attn_dim(), 8);
        assert_eq!(c.geom.kv_dim(), 4);
        assert!(c.geom.engram_sites.is_empty());
        assert_eq!(c.geom.engram_window(), 0);
    }

    #[test]
    fn layout_resolves_canon_positions() {
        let c = Cact::from_bytes(synth(3, true, true)).unwrap();
        let l = c.layout().unwrap();
        assert_eq!(l.embedding, 0);
        assert_eq!(l.layers.len(), 3);
        assert_eq!(l.layers[0].norm_in, 1);
        assert_eq!(l.layers[0].d3, 14);
        assert_eq!(l.layers[1].norm_in, 15);
        assert_eq!(l.mhc.a_pre, 1 + 3 * TENSORS_PER_LAYER);
        assert_eq!(l.mhc.phi_res, l.mhc.a_pre + 8);
        assert_eq!(l.final_norm, l.mhc.a_pre + MHC_TENSORS);
        assert_eq!(l.heads.len(), 2);
        assert_eq!(l.heads[0].code, HEAD_CONTRASTIVE);
        assert_eq!(l.heads[1].code, HEAD_CONFIDENCE);
        assert_eq!(l.tokenizer, Some(c.num_tensors() - 1));
        assert!(l.head(HEAD_CONFIDENCE).is_some());
    }

    #[test]
    fn layout_without_heads_or_tokenizer() {
        let c = Cact::from_bytes(synth(2, false, false)).unwrap();
        let l = c.layout().unwrap();
        assert!(l.heads.is_empty());
        assert!(l.head_manifest.is_none());
        assert!(l.tokenizer.is_none());
        assert_eq!(l.final_norm, c.num_tensors() - 1);
    }

    #[test]
    fn heads_present_but_no_tokenizer() {
        let c = Cact::from_bytes(synth(2, true, false)).unwrap();
        let l = c.layout().unwrap();
        assert_eq!(l.heads.len(), 2);
        assert!(l.tokenizer.is_none());
    }

    #[test]
    fn rejects_bad_tag() {
        let mut b = synth(2, false, false);
        b[0] ^= 0xFF;
        assert!(matches!(Cact::from_bytes(b), Err(CactError::BadTag(_))));
    }

    #[test]
    fn rejects_truncated_file() {
        let b = synth(2, false, false);
        let n = b.len();
        assert!(matches!(
            Cact::from_bytes(b[..n / 2].to_vec()),
            Err(CactError::BlobOutOfRange { .. }) | Err(CactError::TooShort { .. })
        ));
    }

    #[test]
    fn rejects_shape_that_contradicts_geometry() {
        let b = synth(2, false, false);
        let mut c = Cact::from_bytes(b).unwrap();
        // Corrupt the recorded d_model of layer 0's norm_in.
        c.records[1].shape[0] += 1;
        assert!(matches!(c.layout(), Err(CactError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_inconsistent_trailing_tensor_count() {
        let mut b = synth(2, true, false);
        // Claim one fewer tensor: 6 trailing instead of 7 -> not 1 + 3h.
        let n = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        b[4..8].copy_from_slice(&(n - 1).to_le_bytes());
        let c = Cact::from_bytes(b).unwrap();
        assert!(matches!(c.layout(), Err(CactError::BadHeadManifest { .. })));
    }

    #[test]
    fn floats_decode_fp16() {
        let c = Cact::from_bytes(synth(1, false, false)).unwrap();
        let v = c.floats(c.layout().unwrap().layers[0].norm_in).unwrap();
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], 1.0); // 0x3C00
    }

    #[test]
    fn dtype_mismatch_is_reported() {
        let c = Cact::from_bytes(synth(1, false, true)).unwrap();
        let tok = c.layout().unwrap().tokenizer.unwrap();
        assert!(matches!(
            c.floats(tok),
            Err(CactError::DtypeMismatch { .. })
        ));
        assert!(matches!(c.cq(0), Err(CactError::DtypeMismatch { .. })));
        assert_eq!(c.raw_tensor(tok).unwrap().len(), 16);
    }

    // ── Needle 3 container ───────────────────────────────────────────────

    /// A minimal v3 header with the geometry the shipped container declares.
    fn v3_header_words() -> Vec<u32> {
        let mut h = vec![0u32; 49];
        h[0] = TAG_V3;
        h[1] = 581; // num_tensors
        h[2] = 28; // codebook_len
        h[3] = 256; // kv_window
        h[4] = 8; // kv_bits
        h[5] = 8192; // vocab
        h[6] = 8192; // out_vocab
        h[7] = 768; // d_model
        h[8] = 12; // num_heads
        h[9] = 2; // num_kv_heads
        h[10] = 20; // num_layers
        h[11] = 48; // qk_head_dim
        h[12] = 64; // v_head_dim
        h[13] = 8192; // max_seq_len
        h[14] = 1024; // hada_n
        h[15] = 4; // mhc_lanes
        h[16] = 1024; // sliding_window
        h[17] = (1 << 4) | (1 << 9) | (1 << 14) | (1 << 19);
        h[19] = 3; // qkv_conv_taps
        h[20] = 18432; // engram_slots
        h[21] = 128; // engram_sub_dim
        h[22] = 6; // num_engram_tables
        h[23] = 4; // engram_conv_taps
        h[24] = 3; // engram_conv_dilation
        h[25] = 0; // engram_seed_heads
        h[26] = 2; // num_engram_orders
        h[27] = 2;
        h[28] = 3;
        h[31] = 5; // num_engram_sites
        h[32] = 3;
        h[33] = 7;
        h[34] = 11;
        h[35] = 15;
        h[36] = 19;
        h[48] = 100_000.0f32.to_bits();
        h
    }

    #[test]
    fn v3_header_parses_every_field() {
        let g = CactV3Geometry::from_words(&v3_header_words()).expect("header should parse");
        assert_eq!(g.d_model, 768);
        assert_eq!(g.num_heads, 12);
        assert_eq!(g.num_kv_heads, 2);
        assert_eq!(g.qk_head_dim, 48);
        assert_eq!(g.v_head_dim, 64);
        assert_eq!(g.num_layers, 20);
        assert_eq!(g.out_vocab, 8192);
        assert_eq!(g.sliding_window, 1024);
        assert_eq!(g.qkv_conv_taps, 3);
        assert_eq!(g.engram_slots, 18432);
        assert_eq!(g.num_engram_tables, 6);
        assert_eq!(g.engram_orders, vec![2, 3]);
        assert_eq!(g.engram_sites, vec![3, 7, 11, 15, 19]);
        assert_eq!(g.global_layers(), vec![4, 9, 14, 19]);
        assert_eq!(g.rope_theta, 100_000.0);
    }

    #[test]
    fn v3_global_mask_spans_both_words() {
        let mut h = v3_header_words();
        h[10] = 40; // 40 layers, so bits land in the high word too
        h[17] = 1 << 4;
        h[18] = 1 << 3; // layer 35
        let g = CactV3Geometry::from_words(&h).expect("header should parse");
        assert_eq!(g.global_layers(), vec![4, 35]);
        assert!(g.is_global(4));
        assert!(g.is_global(35));
        assert!(!g.is_global(5));
    }

    #[test]
    fn v3_rejects_the_v2_tag() {
        let mut h = v3_header_words();
        h[0] = TAG;
        assert!(matches!(
            CactV3Geometry::from_words(&h),
            Err(CactError::BadTag(_))
        ));
    }

    #[test]
    fn v2_loader_rejects_a_v3_tag() {
        let mut b = synth(3, true, true);
        b[0..4].copy_from_slice(&TAG_V3.to_le_bytes());
        assert!(matches!(Cact::from_bytes(b), Err(CactError::BadTag(_))));
    }
}
