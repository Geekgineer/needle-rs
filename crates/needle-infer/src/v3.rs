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
