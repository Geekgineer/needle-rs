//! Needle 3.
//!
//! Sits beside [`crate::v2`] rather than replacing it: the two share no
//! weights, no container layout and no forward pass, only primitives.

pub mod attention;
pub mod config;
pub mod kernels;

pub use attention::{attend, causal_depthwise_conv, norm_and_rope, AttnDims};
pub use config::{V3Config, V3Engram};
pub use kernels::{hada_blocks, hadamard_mlp, kron_apply, HadaMlp, HadaPerms};
