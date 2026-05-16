//! WASM bindings for Needle.
//! Compiled with `wasm-pack build --target web` (or `--target nodejs`).
//! Exposes a simple JS API: NeedleWasm.load(weights_bytes, vocab_text) → run(query, tools).

// WASM-specific imports and #[wasm_bindgen] attributes will be added
// once wasm-bindgen is integrated. For now this is a structural placeholder
// that compiles for native and will be wired up for WASM as Tier 2.

use needle_infer::NeedleEngine;

/// WASM handle (will become #[wasm_bindgen] struct).
pub struct NeedleWasm {
    // engine: NeedleEngine,  // uncomment when WASM I/O (from_bytes) is implemented
}
