#![no_std]
#![allow(clippy::excessive_precision)]

extern crate alloc;

pub mod math;
pub mod config;
pub mod quant;
pub mod ops;
pub mod rope;
pub mod norm;
pub mod attn;
pub mod ffn;
pub mod layers;
pub mod model;

pub use config::TransformerConfig;
