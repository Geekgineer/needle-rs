//! Needle 3.
//!
//! Sits beside [`crate::v2`] rather than replacing it: the two share no
//! weights, no container layout and no forward pass, only primitives.

pub mod attention;
pub mod config;
pub mod engram;
pub mod kernels;
pub mod mhc;
pub mod model;

pub use attention::{attend, causal_depthwise_conv, norm_and_rope, AttnDims};
pub use config::{V3Config, V3Engram};
pub use engram::{engram_indices, ngram_valid, EngramDims};
pub use kernels::{hada_blocks, hadamard_mlp, kron_apply, HadaMlp, HadaPerms};
pub use mhc::{mix_down, post_off, pre_off, scatter_up, MhcLayer};
pub use model::{V3EngramSite, V3Layer, V3Mhc, V3Model};
