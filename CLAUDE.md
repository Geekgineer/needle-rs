# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

Ultra-fast, zero-dependency Rust inference engine for the Needle SAN (Simple Attention Network) — a 26M parameter encoder-decoder transformer for single-shot tool/function calling. Goals: hit the memory-bandwidth ceiling on native (AVX2/NEON), WASM SIMD128, and scalar fallback, with a sub-1MB binary and sub-14MB weight file.

## Common Commands

```bash
# Check all crates compile
cargo check

# Run tests
cargo test -p needle-core

# Release build
cargo build --release

# Export Python checkpoint to SafeTensors (run from needle/ Python env)
python tools/export.py --checkpoint needle/checkpoints/needle.pkl --output-dir weights/

# Run inference
./target/release/needle-rs weights/needle.safetensors weights/vocab.txt "What's the weather?" '[{"name":"get_weather","description":"...","parameters":{}}]'
```

## Workspace Structure

```
crates/
  needle-core/    # no_std compute kernels (quant, attn, norm, rope, ffn, model)
  needle-infer/   # std inference engine (safetensors, tokenizer, constrained decode, engine)
  needle-c/       # C ABI cdylib (FFI for Python/Go/Swift/etc.)
  needle-wasm/    # WASM bindings (Tier 2, structural placeholder)
  needle-cli/     # CLI binary
tools/
  export.py       # Converts needle.pkl → needle.safetensors + vocab.txt
```

## Architecture

### needle-core (no_std + alloc)

All compute — no panic in release, no heap allocation in the hot path except pre-allocated buffers.

- **`quant.rs`**: INT4 symmetric group-wise quantization (`group_size=32`, scale=`max(abs)/7.0`, clip `[-8,7]`). Packed nibbles: low nibble = even row, high nibble = odd row, per output-feature column. `QuantizedWeight::matvec` dequantizes on the fly (never materializes full weight matrix). Must stay bit-for-bit identical to Python `quantize.py`.
- **`norm.rs`**: ZCRMSNorm — `(1 + γ) * x / RMS(x)`, γ initialized to 0.
- **`rope.rs`**: RoPE with precomputed cos/sin table. Applied to Q and K (interleaved, not concatenated halves).
- **`attn.rs`**: GQA (8Q/4KV, repeat=2). `KvCache` stores K/V for incremental decode. Self-attn full (encoder), self-attn incremental (decoder), cross-attn incremental (decoder).
- **`layers.rs`**: Gated residual `x += sigmoid(gate) * sublayer(norm(x))`. Encoder = self-attn only. Decoder = self-attn + cross-attn + optional FFN.
- **`model.rs`**: `NeedleModel` — tied embedding (LM head = embedding^T), encode(), decode_step().
- **`math.rs`**: Thin wrappers over `libm` for all transcendentals (exp, sin, cos, sqrt, powf, tanh, round).

### needle-infer (std)

- **`safetensors.rs`**: Self-contained SafeTensors reader. No external deps. Handles F32/BF16/F16/I8/I4.
- **`tokenizer.rs`**: SentencePiece vocabulary loader (text format exported by `tools/export.py`). Special IDs: PAD=0, EOS=1, BOS=2, UNK=3, TOOL_CALL=4, TOOLS=5. `to_snake_case()` matches Python exactly.
- **`constrained.rs`**: Character-level trie for constrained decoding of tool names and argument keys.
- **`engine.rs`**: `NeedleEngine::load()` + `run()`. SafeTensors tensor naming convention defined here — must match `tools/export.py`.

### Tensor naming convention (engine.rs ↔ export.py)

| Tensor | Name |
|--------|------|
| Embedding | `embedding` |
| Final norm | `final_norm` |
| Encoder layer i self-attn Q | `encoder.{i}.self_attn.wq` |
| Encoder layer i norm | `encoder.{i}.norm` |
| Encoder layer i gate | `encoder.{i}.self_attn_gate` |
| Decoder layer i cross-attn K | `decoder.{i}.cross_attn.wk` |
| Quantized weight data | `<name>` (I4 dtype) |
| Quantized weight scales | `<name>.scale` (F32) |

## Weight Format

SafeTensors: 8-byte LE header length + JSON header + raw tensor data.  
Quantized kernels stored as packed INT4 nibbles (`I4` dtype) with a separate `.scale` tensor.  
Non-kernel params stored as BF16.

## INT4 Quantization (critical — must match Python exactly)

```
gs = min(32, in_feat)
pad in_feat to multiple of gs
num_groups = in_padded / gs
scale[g, o] = max(abs(w[g*gs:(g+1)*gs, o])) / 7.0, clamped >= 1e-8
w_q = clip(round(w / scale), -8, 7)
pack: byte[o * (in_padded/2) + pair] = lo_nibble(row 2*pair) | hi_nibble(row 2*pair+1)
```

Applied only to Dense kernels (wq, wk, wv, wo). 3D scan-stacked params: quantize each slice.

## Key Invariants

- No `std` in `needle-core` — keeps it `no_std` compatible for WASM/embedded.
- `libm` is the only external dependency in `needle-core`.
- Encoder runs once; its K/V is stored in `enc_kv_caches` and reused for all decoder steps.
- Decoder self-attn cache grows one slot per step; cross-attn cache is static (encoder output).
- Tied embedding: `logits[v] = dot(hidden, embedding[v])` — no separate output projection weights.

## Next Steps (not yet implemented)

1. **Full SentencePiece tokenization**: wire up `sentencepiece-rs` or FFI to sentencepiece C lib
2. **SIMD kernels**: AVX2 INT4 matmul in `needle-core` (feature-gated, `#[cfg(target_feature = "avx2")]`)
3. **Constraint state machine**: complete `update_constraint_state` in `engine.rs` for arg-key constraints
4. **WASM bindings**: add `wasm-bindgen` to `needle-wasm`, implement `from_bytes()` loader
5. **Parity test suite**: 500-example Python-vs-Rust token comparison
6. **export.py**: fix actual Flax param path mapping (the `map_flax_key` function is a stub)
