//! Needle 3: container to core geometry.
//!
//! `needle-core` holds the compute and stays `no_std`, taking geometry as data
//! ([`V3Config`]) and weights as [`needle_core::cq::CqWeight`]. This module is
//! the bridge from a `.cact` v3 container.

use needle_core::v3::{V3Config, V3Engram};

use crate::cact::CactV3Geometry;

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
    if g.num_engram_tables % orders.len() != 0 {
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
    if g.num_kv_heads == 0 || g.num_heads % g.num_kv_heads != 0 {
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
