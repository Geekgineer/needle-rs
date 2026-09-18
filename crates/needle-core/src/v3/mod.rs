//! Needle 3.
//!
//! Sits beside [`crate::v2`] rather than replacing it: the two share no
//! weights, no container layout and no forward pass, only primitives.

pub mod attention;
pub mod cache;
pub mod config;
pub mod engram;
pub mod heads;
pub mod kernels;
pub mod mhc;
pub mod model;

pub use attention::{
    attend, attend_step, causal_depthwise_conv, norm_and_rope, AttnDims, KvStore, Ring,
};
pub use cache::{fake_quant_vec, quantize_rows, KvPrecision, Qkv, V3Cache};
pub use config::{V3Config, V3Engram};
pub use engram::{engram_indices, ngram_valid, EngramDims};
pub use heads::{ProbeHead, HEAD_CONFIDENCE, HEAD_EMBEDDING, HEAD_ROUTER};
pub use kernels::{hada_blocks, hadamard_mlp, kron_apply, HadaMlp, HadaPerms};
pub use mhc::{mix_down, post_off, pre_off, scatter_up, MhcLayer};
pub use model::{V3EngramSite, V3Layer, V3Mhc, V3Model};
