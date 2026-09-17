//! Needle 3.
//!
//! Sits beside [`crate::v2`] rather than replacing it: the two share no
//! weights, no container layout and no forward pass, only primitives.

pub mod config;

pub use config::{V3Config, V3Engram};
