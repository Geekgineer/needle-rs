//! Needle 3: container to core geometry.
//!
//! `needle-core` holds the compute and stays `no_std`, taking geometry as data
//! ([`V3Config`]) and weights as [`needle_core::cq::CqWeight`]. This module is
//! the bridge from a `.cact` v3 container.

use needle_core::v3::kernels::{HadaMlp, HadaPerms};
extern crate alloc;

use needle_core::cq::CqWeight;
use needle_core::v3::heads::{ProbeHead, HEAD_CONFIDENCE};
use needle_core::v3::ladder_layer_indices;
use needle_core::v3::{V3Config, V3Engram, V3EngramSite, V3Layer, V3Mhc, V3Model};

use crate::cact::{CactV3, CactV3Geometry};

/// Derive the core geometry from what a v3 container declares.
///
/// Everything is read from the header. The one derived value is the Engram
/// head count: upstream computes `heads = d_model / (len(orders) * sub_dim)`
/// and stores `num_tables = len(orders) * heads`, so the head count is
/// recovered by dividing back out. Asserting that relationship here means a
/// container whose table count disagrees with its own geometry fails at load
/// rather than silently indexing the wrong hash table.
pub fn config_from_geometry(g: &CactV3Geometry) -> Result<V3Config, V3GeometryError> {
    let orders = g.engram_orders.clone();
    if orders.is_empty() {
        return Err(V3GeometryError::NoEngramOrders);
    }
    if !g.num_engram_tables.is_multiple_of(orders.len()) {
        return Err(V3GeometryError::TableCountNotDivisible {
            tables: g.num_engram_tables,
            orders: orders.len(),
        });
    }
    let heads = g.num_engram_tables / orders.len();
    let expect_sub_dim = g.d_model / (orders.len() * heads);
    if expect_sub_dim != g.engram_sub_dim {
        return Err(V3GeometryError::SubDimMismatch {
            declared: g.engram_sub_dim,
            derived: expect_sub_dim,
        });
    }
    if g.num_kv_heads == 0 || !g.num_heads.is_multiple_of(g.num_kv_heads) {
        return Err(V3GeometryError::HeadsNotDivisible {
            heads: g.num_heads,
            kv_heads: g.num_kv_heads,
        });
    }

    Ok(V3Config {
        vocab_size: g.vocab_size,
        out_vocab: g.out_vocab,
        d_model: g.d_model,
        num_heads: g.num_heads,
        num_kv_heads: g.num_kv_heads,
        num_layers: g.num_layers,
        qk_head_dim: g.qk_head_dim,
        v_head_dim: g.v_head_dim,
        max_seq_len: g.max_seq_len,
        hada_n: g.hada_n,
        mhc_lanes: g.mhc_lanes,
        rope_theta: g.rope_theta,
        sliding_window: g.sliding_window,
        global_layers: g.global_layers(),
        qkv_conv_taps: g.qkv_conv_taps,
        engram: V3Engram {
            orders,
            heads,
            slots: g.engram_slots,
            sub_dim: g.engram_sub_dim,
            sites: g.engram_sites.clone(),
            conv_taps: g.engram_conv_taps,
            conv_dilation: g.engram_conv_dilation,
            seed_heads: g.engram_seed_heads,
        },
    })
}

/// A container whose header is self-inconsistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V3GeometryError {
    NoEngramOrders,
    TableCountNotDivisible { tables: usize, orders: usize },
    SubDimMismatch { declared: usize, derived: usize },
    HeadsNotDivisible { heads: usize, kv_heads: usize },
}

impl core::fmt::Display for V3GeometryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEngramOrders => write!(f, "container declares no engram orders"),
            Self::TableCountNotDivisible { tables, orders } => write!(
                f,
                "engram tables {tables} is not a multiple of {orders} orders, \
                 so the per-order head count is undefined"
            ),
            Self::SubDimMismatch { declared, derived } => write!(
                f,
                "engram sub_dim {declared} disagrees with d_model / (orders * heads) \
                 = {derived}"
            ),
            Self::HeadsNotDivisible { heads, kv_heads } => write!(
                f,
                "{heads} query heads do not divide into {kv_heads} kv heads"
            ),
        }
    }
}

impl std::error::Error for V3GeometryError {}

/// Directory indices for one v3 layer, in canon order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3LayerIdx {
    pub norm_in: usize,
    pub q_proj: usize,
    pub k_proj: usize,
    pub v_proj: usize,
    /// Causal depthwise conv taps over Q, K and V. `None` when the model
    /// declares `qkv_conv_taps == 0`.
    pub qkv_taps: Option<[usize; 3]>,
    pub q_norm: usize,
    pub k_norm: usize,
    pub gate_proj: usize,
    pub out_proj: usize,
    pub post_norm: usize,
    pub attn_gate: usize,
    pub pre_hada: usize,
    /// HadamardMLP: d1, d2, b2, d3, d4, w1a, w1b, w2a, w2b, w3a, w3b,
    /// cond_v, cond_u — in that order.
    pub mlp: [usize; 13],
}

/// Directory indices for the mHC lane-mixing parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3MhcIdx {
    /// a_pre, a_post, a_res, b_pre, b_post, b_res.
    pub scalars: [usize; 6],
    /// phi_pre, phi_post, phi_res. `phi_res` carries `lanes^2` on its
    /// trailing axis where the others carry `lanes`.
    pub phi: [usize; 3],
}

/// Directory indices for one Engram site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3EngramIdx {
    pub tables: usize,
    pub key_proj: usize,
    pub value_proj: usize,
    pub taps: usize,
}

/// Where every tensor lives in a v3 container's nameless directory.
///
/// This is the canon, and it is the only thing standing between a correct
/// container and confidently wrong logits: the directory carries no names, so
/// nothing but this ordering says which slot is `q_proj` for layer 7. It
/// mirrors `_tensors` in upstream `export.py` exactly, and every slot's shape
/// is validated against the declared geometry, so a layout drift fails at load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V3Layout {
    pub embedding: usize,
    pub layers: Vec<V3LayerIdx>,
    pub mhc: V3MhcIdx,
    /// The two derived Hadamard permutations, stored as FP32.
    pub hada_perms: [usize; 2],
    pub engrams: Vec<V3EngramIdx>,
    pub final_norm: usize,
    /// `heads.manifest`, present only when probe heads were exported.
    pub head_manifest: Option<usize>,
    /// Remaining head tensors, positional after the manifest.
    pub head_tensors: Vec<usize>,
    pub tokenizer: Option<usize>,
}

/// A container whose directory does not match its own header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V3LayoutError {
    /// The directory ran out before the canon did.
    TooFewTensors { need: usize, got: usize },
    /// A slot's shape contradicts the declared geometry.
    BadShape {
        index: usize,
        what: &'static str,
        want: [usize; 2],
        got: [usize; 2],
    },
}

impl core::fmt::Display for V3LayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewTensors { need, got } => write!(
                f,
                "canon needs at least {need} tensors, directory has {got}"
            ),
            Self::BadShape {
                index,
                what,
                want,
                got,
            } => write!(
                f,
                "slot {index} ({what}): want {}x{}, got {}x{}",
                want[0], want[1], got[0], got[1]
            ),
        }
    }
}

impl std::error::Error for V3LayoutError {}

/// Walks the canon, so an exhausted directory reports which slot ran out.
struct Cursor {
    i: usize,
    n: usize,
}

impl Cursor {
    fn next(&mut self) -> Result<usize, V3LayoutError> {
        if self.i >= self.n {
            return Err(V3LayoutError::TooFewTensors {
                need: self.i + 1,
                got: self.n,
            });
        }
        let idx = self.i;
        self.i += 1;
        Ok(idx)
    }
}

impl V3Layout {
    /// Walk the canon for `cfg`, validating shapes against `records`.
    pub fn derive(cfg: &V3Config, records: &[crate::cact::Record]) -> Result<Self, V3LayoutError> {
        let n = records.len();
        let mut cur = Cursor { i: 0, n };
        macro_rules! take {
            () => {
                cur.next()?
            };
        }

        let embedding = take!();

        let taps_n = cfg.qkv_conv_taps;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let norm_in = take!();
            let q_proj = take!();
            let k_proj = take!();
            let v_proj = take!();
            let qkv_taps = if taps_n > 0 {
                Some([take!(), take!(), take!()])
            } else {
                None
            };
            let q_norm = take!();
            let k_norm = take!();
            let gate_proj = take!();
            let out_proj = take!();
            let post_norm = take!();
            let attn_gate = take!();
            let pre_hada = take!();
            let mut mlp = [0usize; 13];
            for slot in mlp.iter_mut() {
                *slot = take!();
            }
            layers.push(V3LayerIdx {
                norm_in,
                q_proj,
                k_proj,
                v_proj,
                qkv_taps,
                q_norm,
                k_norm,
                gate_proj,
                out_proj,
                post_norm,
                attn_gate,
                pre_hada,
                mlp,
            });
        }

        let mut scalars = [0usize; 6];
        for slot in scalars.iter_mut() {
            *slot = take!();
        }
        let mut phi = [0usize; 3];
        for slot in phi.iter_mut() {
            *slot = take!();
        }
        let hada_perms = [take!(), take!()];

        let mut engrams = Vec::with_capacity(cfg.engram.sites.len());
        for _ in 0..cfg.engram.sites.len() {
            engrams.push(V3EngramIdx {
                tables: take!(),
                key_proj: take!(),
                value_proj: take!(),
                taps: take!(),
            });
        }

        let final_norm = take!();

        // The tokenizer is the single RAW record and rides at the end.
        let tokenizer = records.iter().position(|r| r.dtype == crate::cact::DT_RAW);
        let head_end = tokenizer.unwrap_or(n);
        let head_manifest = if cur.i < head_end {
            Some(take!())
        } else {
            None
        };
        let head_tensors: Vec<usize> = (cur.i..head_end).collect();

        let layout = Self {
            embedding,
            layers,
            mhc: V3MhcIdx { scalars, phi },
            hada_perms,
            engrams,
            final_norm,
            head_manifest,
            head_tensors,
            tokenizer,
        };
        layout.validate(cfg, records)?;
        Ok(layout)
    }

    /// Check the slots whose shapes the geometry fully determines. A correct
    /// tensor in the wrong slot is exactly what this catches.
    fn validate(
        &self,
        cfg: &V3Config,
        records: &[crate::cact::Record],
    ) -> Result<(), V3LayoutError> {
        let check =
            |index: usize, what: &'static str, want: [usize; 2]| -> Result<(), V3LayoutError> {
                let r = &records[index];
                let got = [r.shape[0], r.shape[1]];
                if got != want {
                    return Err(V3LayoutError::BadShape {
                        index,
                        what,
                        want,
                        got,
                    });
                }
                Ok(())
            };

        check(self.embedding, "embedding", [cfg.vocab_size, cfg.d_model])?;
        for l in &self.layers {
            check(l.q_proj, "q_proj", [cfg.q_dim(), cfg.d_model])?;
            check(l.k_proj, "k_proj", [cfg.k_dim(), cfg.d_model])?;
            check(l.v_proj, "v_proj", [cfg.v_dim(), cfg.d_model])?;
            // The gate is elementwise over the attention output, not per head:
            // upstream is Dense(out_dim) with out_dim = num_heads * v_head_dim.
            check(l.gate_proj, "gate_proj", [cfg.attn_out_dim(), cfg.d_model])?;
            check(l.out_proj, "out_proj", [cfg.d_model, cfg.attn_out_dim()])?;
        }
        for e in &self.engrams {
            check(
                e.tables,
                "engram tables",
                [
                    cfg.engram.num_tables() * cfg.engram.slots,
                    cfg.engram.sub_dim,
                ],
            )?;
            check(e.key_proj, "engram key_proj", [cfg.d_model, cfg.d_model])?;
            check(
                e.value_proj,
                "engram value_proj",
                [cfg.d_model, cfg.d_model],
            )?;
        }
        Ok(())
    }
}

/// Build a [`V3Model`] from a loaded container.
///
/// Weights stay packed: `CqWeight` holds the 2- and 4-bit tensors and is
/// matvec'd directly, so the 35 MB container never expands to a dense copy.
/// Only the FP16/FP32 vectors — norms, gates, diagonals, Kronecker factors and
/// the Hadamard permutations — are decoded to `f32`.
/// Build one layer from its canon slots.
///
/// Shared by the full-depth loader and the ladder slice so the two cannot drift
/// apart: a rung differs by *which* layers it takes, never by how one is read.
fn layer_from_layout(cact: &CactV3, l: &V3LayerIdx) -> Result<V3Layer, V3LoadError> {
    let (q_taps, k_taps, v_taps) = match l.qkv_taps {
        Some([q, k, v]) => (cact.floats(q)?, cact.floats(k)?, cact.floats(v)?),
        None => (Vec::new(), Vec::new(), Vec::new()),
    };
    let m = l.mlp;
    Ok(V3Layer {
        norm_in: cact.floats(l.norm_in)?,
        q_proj: cact.cq(l.q_proj)?,
        k_proj: cact.cq(l.k_proj)?,
        v_proj: cact.cq(l.v_proj)?,
        q_taps,
        k_taps,
        v_taps,
        q_norm: cact.floats(l.q_norm)?,
        k_norm: cact.floats(l.k_norm)?,
        gate_proj: cact.cq(l.gate_proj)?,
        out_proj: cact.cq(l.out_proj)?,
        post_norm: cact.floats(l.post_norm)?,
        attn_gate: *cact
            .floats(l.attn_gate)?
            .first()
            .ok_or(V3LoadError::EmptyScalar("attn_gate"))?,
        pre_hada: cact.floats(l.pre_hada)?,
        // Canon order: d1, d2, b2, d3, d4, w1a, w1b, w2a, w2b, w3a, w3b,
        // cond_v, cond_u.
        mlp: HadaMlp {
            d1: cact.floats(m[0])?,
            d2: cact.floats(m[1])?,
            b2: cact.floats(m[2])?,
            d3: cact.floats(m[3])?,
            d4: cact.floats(m[4])?,
            w1: (cact.floats(m[5])?, cact.floats(m[6])?),
            w2: (cact.floats(m[7])?, cact.floats(m[8])?),
            w3: (cact.floats(m[9])?, cact.floats(m[10])?),
            cond_v: cact.floats(m[11])?,
            cond_u: cact.floats(m[12])?,
            cond_rank: cact.record(m[11]).shape[1],
        },
    })
}

pub fn model_from_cact(cact: &CactV3) -> Result<V3Model, V3LoadError> {
    let cfg = config_from_geometry(&cact.geom)?;
    let layout = V3Layout::derive(&cfg, cact.records())?;

    let embedding = cact.cq(layout.embedding)?;

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for l in &layout.layers {
        layers.push(layer_from_layout(cact, l)?);
    }

    let sc = layout.mhc.scalars;
    let phi = layout.mhc.phi;
    let mhc = V3Mhc {
        a_pre: cact.floats(sc[0])?,
        a_post: cact.floats(sc[1])?,
        a_res: cact.floats(sc[2])?,
        b_pre: cact.floats(sc[3])?,
        b_post: cact.floats(sc[4])?,
        b_res: cact.floats(sc[5])?,
        phi_pre: cact.cq(phi[0])?,
        phi_post: cact.cq(phi[1])?,
        phi_res: cact.cq(phi[2])?,
    };

    let mut engrams = Vec::with_capacity(layout.engrams.len());
    for e in &layout.engrams {
        engrams.push(V3EngramSite {
            tables: cact.cq(e.tables)?,
            key_proj: cact.cq(e.key_proj)?,
            value_proj: cact.cq(e.value_proj)?,
            taps: cact.floats(e.taps)?,
        });
    }

    // The permutations ride as FP32 because upstream derives them from a
    // seeded RNG no runtime can reproduce.
    let to_perm = |v: Vec<f32>| -> Vec<u32> { v.into_iter().map(|x| x as u32).collect() };
    let perms = HadaPerms {
        p1: to_perm(cact.floats(layout.hada_perms[0])?),
        p2: to_perm(cact.floats(layout.hada_perms[1])?),
    };

    let final_norm = cact.floats(layout.final_norm)?;
    V3Model::new(cfg, embedding, layers, mhc, engrams, final_norm, perms)
        .map_err(V3LoadError::Shape)
}

/// Load the probe heads a container exports, keyed by head code.
///
/// The manifest lists which heads are present; their tensors follow in canon
/// order, six per head. v3 exports only `confidence`, so `retrieve_tools` has
/// no v3 equivalent — callers must check rather than assume.
pub fn heads_from_cact(cact: &CactV3, cfg: &V3Config) -> Result<Vec<(u8, ProbeHead)>, V3LoadError> {
    let layout = V3Layout::derive(cfg, cact.records())?;
    let Some(manifest_idx) = layout.head_manifest else {
        return Ok(Vec::new());
    };
    let codes: Vec<u8> = cact
        .floats(manifest_idx)?
        .into_iter()
        .map(|c| c as u8)
        .collect();

    let mut out = Vec::new();
    let mut it = layout.head_tensors.iter().copied();
    for code in codes {
        // Canon order per head: probes, gain, query, row_bias, proj, bias.
        let mut next = || -> Result<usize, V3LoadError> {
            it.next().ok_or(V3LoadError::Shape("head tensors ran out"))
        };
        let probes_i = next()?;
        let gain_i = next()?;
        let query_i = next()?;
        let row_bias_i = next()?;
        let proj_i = next()?;
        let bias_i = next()?;
        // The router head carries one extra tensor.
        if code == needle_core::v3::heads::HEAD_ROUTER {
            let _calibration = next()?;
        }

        let probes_rec = cact.record(probes_i);
        let query_rec = cact.record(query_i);
        let gain = cact.floats(gain_i)?;
        let d = probes_rec.shape[1];

        // One cell per layer plus the input embedding — that is what the head
        // reads. `probes` is stored as (cells * probes, d), so the per-cell
        // probe count follows; `gain` is the same product and cross-checks it.
        let cells = cfg.num_layers + 1;
        if !probes_rec.shape[0].is_multiple_of(cells) || gain.len() != probes_rec.shape[0] {
            return Err(V3LoadError::Shape(
                "probe head shapes disagree with num_layers + 1 cells",
            ));
        }
        let probes_per_cell = probes_rec.shape[0] / cells;
        let queries = query_rec.shape[0];

        let head = ProbeHead {
            probes: dense(cact, probes_i)?,
            gain,
            query: dense(cact, query_i)?,
            row_bias: cact.floats(row_bias_i)?,
            proj: dense(cact, proj_i)?,
            bias: cact.floats(bias_i)?,
            cells,
            probes_per_cell,
            queries,
            out_dim: cact.record(proj_i).shape[0],
        };
        debug_assert_eq!(head.probes.len(), cells * probes_per_cell * d);
        out.push((code, head));
    }
    Ok(out)
}

/// Dequantise a CQ tensor. Probe-head matrices are small — 84x768 on the
/// shipped model — so packing them buys nothing and the head reads dense.
fn dense(cact: &CactV3, idx: usize) -> Result<Vec<f32>, V3LoadError> {
    let w = cact.cq(idx)?;
    let mut out = alloc::vec![0.0f32; w.out_feat * w.in_feat];
    w.dequantize_to(&mut out);
    Ok(out)
}

/// The confidence head, if the container exports one.
pub fn confidence_head(cact: &CactV3, cfg: &V3Config) -> Result<Option<ProbeHead>, V3LoadError> {
    Ok(heads_from_cact(cact, cfg)?
        .into_iter()
        .find(|(c, _)| *c == HEAD_CONFIDENCE)
        .map(|(_, h)| h))
}

/// Anything that can go wrong turning a container into a model.
#[derive(Debug)]
pub enum V3LoadError {
    Cact(crate::cact::CactError),
    Geometry(V3GeometryError),
    Layout(V3LayoutError),
    EmptyScalar(&'static str),
    Shape(&'static str),
}

impl From<crate::cact::CactError> for V3LoadError {
    fn from(e: crate::cact::CactError) -> Self {
        Self::Cact(e)
    }
}
impl From<V3GeometryError> for V3LoadError {
    fn from(e: V3GeometryError) -> Self {
        Self::Geometry(e)
    }
}
impl From<V3LayoutError> for V3LoadError {
    fn from(e: V3LayoutError) -> Self {
        Self::Layout(e)
    }
}

impl core::fmt::Display for V3LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cact(e) => write!(f, "container: {e}"),
            Self::Geometry(e) => write!(f, "geometry: {e}"),
            Self::Layout(e) => write!(f, "layout: {e}"),
            Self::EmptyScalar(w) => write!(f, "{w} tensor is empty"),
            Self::Shape(w) => write!(f, "{w}"),
        }
    }
}

impl std::error::Error for V3LoadError {}

// ── Ladder rungs, sliced at load ────────────────────────────────────────────

/// The config a rung of `depth` declares, derived from its parent's.
///
/// Mirrors upstream `ladder_config`: the kept blocks are renumbered
/// contiguously, and the global-attention layers and Engram sites follow them.
/// A container built by `needle build --layers N` carries exactly this, which
/// is what makes a load-time slice and a pre-built rung the same model.
pub fn config_at_depth(parent: &V3Config, depth: usize) -> Option<V3Config> {
    let kept = ladder_layer_indices(parent.num_layers, depth)?;
    let rank = |l: usize| kept.iter().position(|&k| k == l);
    let mut cfg = parent.clone();
    cfg.num_layers = depth;
    cfg.global_layers = parent
        .global_layers
        .iter()
        .filter_map(|&l| rank(l))
        .collect();
    cfg.engram.sites = parent
        .engram
        .sites
        .iter()
        .filter_map(|&l| rank(l))
        .collect();
    Some(cfg)
}

/// Load the `depth`-block rung of a full-depth container.
///
/// Needle 3 is laddered: every depth from 2 to `num_layers` is a trained
/// subnetwork, and upstream ships one by rewriting the container. This does the
/// same slice at load time, so one 35 MB file serves every rung without a
/// second download.
///
/// The slice is exact, not approximate. Per-layer tensors are kept whole;
/// Cactus-Quants packs each row independently, so the mHC lane matrices are cut
/// with [`CqWeight::select_rows`] and keep the codes they were trained with;
/// and the Engram tables of surviving sites are kept entire and renumbered.
/// Nothing is re-quantised.
///
/// Note that a block's mHC lane is `index % lanes`, computed from its position
/// in the *sliced* stack — so a kept block generally moves lane. That is
/// upstream's behaviour, not an oversight: the ladder is trained across sampled
/// depths precisely so the blocks tolerate it.
pub fn model_from_cact_at_depth(cact: &CactV3, depth: usize) -> Result<V3Model, V3LoadError> {
    let parent = config_from_geometry(&cact.geom)?;
    if depth == parent.num_layers {
        return model_from_cact(cact);
    }
    let kept = ladder_layer_indices(parent.num_layers, depth)
        .ok_or(V3LoadError::Shape("depth is outside the ladder"))?;
    let cfg = config_at_depth(&parent, depth).ok_or(V3LoadError::Shape("bad ladder depth"))?;
    let layout = V3Layout::derive(&parent, cact.records())?;
    let n = parent.mhc_lanes;

    let mut layers = Vec::with_capacity(depth);
    for &src in &kept {
        layers.push(layer_from_layout(cact, &layout.layers[src])?);
    }

    // Per-layer mHC scalars follow the kept blocks; the lane matrices are rows
    // `layer * lanes` and `layer * lanes²`.
    let take = |v: &[f32], width: usize| -> Vec<f32> {
        let mut out = Vec::with_capacity(kept.len() * width);
        for &l in &kept {
            out.extend_from_slice(&v[l * width..(l + 1) * width]);
        }
        out
    };
    let rows = |width: usize| -> Vec<usize> {
        kept.iter()
            .flat_map(|&l| l * width..(l + 1) * width)
            .collect()
    };
    let sc = layout.mhc.scalars;
    let phi = layout.mhc.phi;
    let cut = |w: CqWeight, width: usize| -> Result<CqWeight, V3LoadError> {
        w.select_rows(&rows(width))
            .ok_or(V3LoadError::Shape("mHC lane row out of range"))
    };
    let mhc = V3Mhc {
        a_pre: take(&cact.floats(sc[0])?, 1),
        a_post: take(&cact.floats(sc[1])?, 1),
        a_res: take(&cact.floats(sc[2])?, 1),
        b_pre: take(&cact.floats(sc[3])?, n),
        b_post: take(&cact.floats(sc[4])?, n),
        b_res: take(&cact.floats(sc[5])?, n * n),
        phi_pre: cut(cact.cq(phi[0])?, n)?,
        phi_post: cut(cact.cq(phi[1])?, n)?,
        phi_res: cut(cact.cq(phi[2])?, n * n)?,
    };

    // An Engram site survives exactly when its block does.
    let mut engrams = Vec::new();
    for (site, e) in layout.engrams.iter().enumerate() {
        let layer = parent.engram.sites[site];
        if !kept.contains(&layer) {
            continue;
        }
        engrams.push(V3EngramSite {
            tables: cact.cq(e.tables)?,
            key_proj: cact.cq(e.key_proj)?,
            value_proj: cact.cq(e.value_proj)?,
            taps: cact.floats(e.taps)?,
        });
    }

    let to_perm = |v: Vec<f32>| -> Vec<u32> { v.into_iter().map(|x| x as u32).collect() };
    let perms = HadaPerms {
        p1: to_perm(cact.floats(layout.hada_perms[0])?),
        p2: to_perm(cact.floats(layout.hada_perms[1])?),
    };
    V3Model::new(
        cfg,
        cact.cq(layout.embedding)?,
        layers,
        mhc,
        engrams,
        cact.floats(layout.final_norm)?,
        perms,
    )
    .map_err(V3LoadError::Shape)
}

/// The confidence head of a `depth`-block rung.
///
/// Upstream takes rows `(0, *(layer + 1 for layer in selected))` from `probes`
/// and `gain`, and the same rows along axis 1 of `row_bias`; `query`, `proj`
/// and `bias` are shared across depths and untouched. Row 0 is the input
/// embedding cell, which every rung keeps.
///
/// There is no per-depth calibration tensor: one head is trained at full depth
/// and sliced. It grows conservative as depth falls — upstream notes that at
/// two blocks it withholds almost every call — which is behaviour, not drift.
pub fn confidence_head_at_depth(
    cact: &CactV3,
    cfg: &V3Config,
    depth: usize,
) -> Result<Option<ProbeHead>, V3LoadError> {
    let Some(full) = confidence_head(cact, cfg)? else {
        return Ok(None);
    };
    if depth == cfg.num_layers {
        return Ok(Some(full));
    }
    let kept = ladder_layer_indices(cfg.num_layers, depth)
        .ok_or(V3LoadError::Shape("depth is outside the ladder"))?;
    let cells: Vec<usize> = core::iter::once(0)
        .chain(kept.iter().map(|&l| l + 1))
        .collect();
    let (k, q, d) = (full.probes_per_cell, full.queries, cfg.d_model);

    let mut probes = Vec::with_capacity(cells.len() * k * d);
    let mut gain = Vec::with_capacity(cells.len() * k);
    for &c in &cells {
        probes.extend_from_slice(&full.probes[c * k * d..(c + 1) * k * d]);
        gain.extend_from_slice(&full.gain[c * k..(c + 1) * k]);
    }
    // row_bias is (queries, cells, probes): the cell axis is the middle one.
    let m = full.cells * k;
    let mut row_bias = Vec::with_capacity(q * cells.len() * k);
    for qi in 0..q {
        for &c in &cells {
            row_bias.extend_from_slice(&full.row_bias[qi * m + c * k..qi * m + (c + 1) * k]);
        }
    }

    Ok(Some(ProbeHead {
        probes,
        gain,
        row_bias,
        cells: cells.len(),
        ..full
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cact::CactV3;

    const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");

    #[test]
    fn derives_the_shipped_geometry() {
        if !std::path::Path::new(CACT).exists() {
            println!("skipping: no weights/needle3.cact");
            return;
        }
        let c = CactV3::load(CACT).expect("load");
        let cfg = config_from_geometry(&c.geom).expect("geometry should be consistent");

        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.num_layers, 20);
        assert_eq!(cfg.qk_head_dim, 48);
        assert_eq!(cfg.v_head_dim, 64);
        assert_eq!(cfg.q_dim(), 576);
        assert_eq!(cfg.k_dim(), 96);
        assert_eq!(cfg.v_dim(), 128);
        assert_eq!(cfg.global_layers, vec![4, 9, 14, 19]);
        assert_eq!(cfg.qkv_conv_taps, 3);
        assert_eq!(cfg.sliding_window, 1024);

        // The head count upstream derives, recovered from the table count.
        assert_eq!(cfg.engram.heads, 3);
        assert_eq!(cfg.engram.num_tables(), 6);
        assert_eq!(cfg.engram.sites, vec![3, 7, 11, 15, 19]);
        assert_eq!(cfg.engram.orders, vec![2, 3]);

        println!(
            "v3 config from container: {}x{}, engram {} tables x {} slots at {:?}",
            cfg.num_layers,
            cfg.d_model,
            cfg.engram.num_tables(),
            cfg.engram.slots,
            cfg.engram.sites
        );
    }

    #[test]
    fn canon_walk_lands_on_the_right_slots() {
        if !std::path::Path::new(CACT).exists() {
            println!("skipping: no weights/needle3.cact");
            return;
        }
        let c = CactV3::load(CACT).expect("load");
        let cfg = config_from_geometry(&c.geom).expect("geometry");
        let l = V3Layout::derive(&cfg, c.records()).expect("canon should walk cleanly");

        // Anchors verified against the container: 581 records, engram sites
        // end at 571, final_norm 572, heads.manifest 573, tokenizer 580.
        assert_eq!(l.embedding, 0);
        assert_eq!(l.layers.len(), 20);
        assert_eq!(l.engrams.len(), 5);
        assert_eq!(l.final_norm, 572);
        assert_eq!(l.head_manifest, Some(573));
        assert_eq!(l.head_tensors, (574..580).collect::<Vec<_>>());
        assert_eq!(l.tokenizer, Some(580));

        // 27 tensors per layer with conv taps present: the first layer starts
        // right after the embedding, the last ends before the mHC block.
        assert_eq!(l.layers[0].norm_in, 1);
        assert_eq!(l.layers[0].qkv_taps, Some([5, 6, 7]));
        assert_eq!(l.mhc.scalars[0], 541);
        assert_eq!(l.mhc.phi[0], 547);
        assert_eq!(l.hada_perms, [550, 551]);
        assert_eq!(l.engrams[0].tables, 552);

        // The two FP32 records are exactly the Hadamard permutations.
        for idx in l.hada_perms {
            let r = c.record(idx);
            assert_eq!(r.dtype, crate::cact::DT_FP32, "hada perm at {idx}");
            assert_eq!(r.shape[0], cfg.hada_n);
        }

        println!(
            "canon: {} layers x 27 tensors, {} engram sites, head tensors {:?}",
            l.layers.len(),
            l.engrams.len(),
            l.head_tensors
        );
    }

    #[test]
    fn a_shape_in_the_wrong_slot_is_rejected() {
        if !std::path::Path::new(CACT).exists() {
            println!("skipping: no weights/needle3.cact");
            return;
        }
        let c = CactV3::load(CACT).expect("load");
        let mut cfg = config_from_geometry(&c.geom).expect("geometry");
        // Claim one layer fewer: the walk stays in bounds but every slot
        // after it shifts by 27, so the engram tables land on a layer tensor.
        // The shape check must catch that rather than proceeding.
        cfg.num_layers = 19;
        let err = V3Layout::derive(&cfg, c.records()).unwrap_err();
        assert!(
            matches!(err, V3LayoutError::BadShape { .. }),
            "expected a shape rejection, got {err}"
        );
    }

    #[test]
    fn rejects_a_table_count_that_contradicts_the_geometry() {
        if !std::path::Path::new(CACT).exists() {
            println!("skipping: no weights/needle3.cact");
            return;
        }
        let c = CactV3::load(CACT).expect("load");
        let mut g = c.geom.clone();
        g.num_engram_tables = 7; // not a multiple of 2 orders
        assert!(matches!(
            config_from_geometry(&g),
            Err(V3GeometryError::TableCountNotDivisible { .. })
        ));

        let mut g = c.geom.clone();
        g.num_engram_tables = 8; // divisible, but then sub_dim would be 96
        assert!(matches!(
            config_from_geometry(&g),
            Err(V3GeometryError::SubDimMismatch { .. })
        ));
    }
}
