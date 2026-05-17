# Architecture

This document describes the needle-rs inference runtime. For model architecture
details (training, hyperparameters, research decisions), see the
[Needle repository by Cactus Compute](https://github.com/cactus-compute/needle).

---

## Crate layout

```
crates/
  needle-core/    no_std compute kernels
  needle-infer/   std inference engine (builds on needle-core)
  needle-c/       C ABI cdylib + staticlib
  needle-wasm/    WASM bindings (wasm-bindgen)
  needle-cli/     CLI binary
```

The dependency graph is a strict DAG: `needle-core` ← `needle-infer` ← `{needle-c, needle-wasm, needle-cli}`.

---

## needle-core — `no_std` kernels

`needle-core` has `#![no_std]` with `extern crate alloc`. This makes it portable to
embedded targets that lack a system allocator. It has one external dependency: `libm`
(transcendental functions — exp, sin, cos, sqrt). All other arithmetic is standard Rust.

### INT4 quantization (`quant.rs`)

Weights are quantized to 4-bit integers group-wise before storage. Dequantization
happens on the fly during matrix-vector multiplication; the full weight matrix is
never materialized in f32.

**Layout:**
- `group_size = 32`
- Scale per group per output feature: `scale[g, o] = max(|w[g*gs..(g+1)*gs, o]|) / 7.0`
- Clipped to `[-8, 7]` (asymmetric around 0 to use all 4-bit patterns)
- Packed nibbles: `byte[pair * out_feat + o]` = low nibble from row `2*pair`, high nibble from row `2*pair+1`

This layout places output-feature stride in the inner loop, which makes the
AVX2 kernel vectorizable without scatter/gather.

**SIMD kernels:**
- `matvec_avx2`: processes 8 output features per AVX2 lane (256-bit). Uses `_mm256_cvtepu8_epi32` for zero-extension, integer masks for sign-extension, and `_mm256_fmadd_ps` for the final FMA. CPUID-gated at runtime — no `target-cpu=native`.
- `matvec_neon`: processes 8 output features per step using ARMv8 NEON intrinsics (`vld1_u8`, `vshr_n_u8`, `vmovl_s8`, `vcvtq_f32_s32`, `vmlaq_f32`). Unconditional on aarch64.
- `matvec_scalar`: portable fallback used for wasm32 and any other target.

### ZCRMSNorm (`norm.rs`)

```
output = (1 + γ) * x / RMS(x)
RMS(x) = sqrt(mean(x²) + ε)
```

`γ` is a per-element learned vector initialized to zero (so initially this is plain RMSNorm).
Matches Python `ZCRMSNorm` in `architecture.py` exactly.

### RoPE (`rope.rs`)

Rotary position embeddings with precomputed cos/sin table. Applied to Q and K
in the encoder's self-attention and the decoder's self-attention. **Not applied
to cross-attention** (Python passes `rope=None` there).

The rotation uses split-half convention:
```
q_rot[0..d/2] = q[0..d/2] * cos - q[d/2..d] * sin
q_rot[d/2..d] = q[d/2..d] * cos + q[0..d/2] * sin
```

### GQA attention (`attn.rs`)

8 query heads / 4 KV heads (repeat=2). Three variants:

1. `self_attn_full`: encoder, runs over the full sequence, no KV cache
2. `self_attn_incremental`: decoder self-attention, one token at a time, appends to `KvCache`
3. `cross_attn_incremental`: decoder cross-attention, reads from pre-filled encoder `KvCache` (no RoPE)

`KvCache` stores K and V concatenated per head in a flat `Vec<f32>`.

### Gated residual (`layers.rs`)

```
x += sigmoid(gate) * sublayer(norm(x))
```

`gate` is a scalar learned parameter. Applied to self-attention, cross-attention,
and optionally FFN in each layer.

### SAN model (`model.rs`)

Encoder: 12 layers of `encoder_layer_forward` (self-attention only) → final norm → cross-attn KV projection.

Decoder: 8 layers of `decoder_layer_forward` (self-attn + cross-attn + optional FFN) → final norm → LM head.

Tied embedding: `logits[v] = dot(hidden, embedding[v])`. No separate output projection matrix.

---

## needle-infer — inference engine

### SafeTensors reader (`safetensors.rs`)

Self-contained parser: 8-byte LE header length + JSON metadata + raw data blobs.
Handles `F32`, `BF16`, `F16`, `I8`, and a custom `I4` dtype (packed nibbles).
No external parsing dependency.

### BPE tokenizer (`tokenizer.rs`)

Loads SentencePiece vocabulary from a text file (`piece TAB score` per line).
Implements the correct iterative-merge algorithm (not greedy-longest-match).
Special IDs: PAD=0, EOS=1, BOS=2, UNK=3, TOOL_CALL=4, TOOLS=5.

### Constrained decoder (`constrained.rs`)

Ensures the model's output is always a valid JSON tool call. Two components:

**1. Character-level trie** — built from the tool names and argument keys at inference time.
Restricts which tokens can start a tool name or argument key.

**2. JSON state machine** — three states:

| State | Description |
|---|---|
| `Free` | Before the opening `{` |
| `InFunctionName` | Inside `"name": "..."` |
| `InArgumentKey` | Inside `"arguments": {"key": ...}` |

At each decode step, invalid-prefix tokens are masked to `-inf` before argmax.

**JSON Schema support:** The Rust decoder handles both formats:
- Flat: `{"location": {"type": "string"}}`
- JSON Schema: `{"type": "object", "properties": {"location": {...}}}`

The Python reference only handles the flat format; JSON Schema input causes it to
insert `"properties"` as a valid argument key. The Rust decoder checks for a nested
`"properties"` key first. Every OpenAI-compatible tool definition uses JSON Schema format.

### Engine (`engine.rs`)

`NeedleEngine::load` reads the SafeTensors file, extracts config from `__metadata__`,
allocates KV caches, and builds the model. The config (d_model, num_heads, etc.) is
embedded in the weights file — no separate config file needed.

`run_impl` implements the full Python-equivalent `generate()`:
1. Normalize tool names to snake_case (for encoder input)
2. Tokenize `[TOOLS] tools_json [TOOLS] [BOS] query` → truncate to `max_enc_len`
3. Encode → fill cross-attn KV caches
4. Decode greedily with constrained decoding until EOS
5. Strip `<tool_call>` prefix
6. Restore original tool names

**Contrastive head:** When `contrastive_proj_kernel` is present in the weights,
the engine loads a two-layer MLP (ReLU hidden, no final activation) for embedding
queries and tool descriptions. Output is L2-normalized. Used by `encode_contrastive`
and `retrieve_tools`.

---

## WASM build details

Target: `wasm32-unknown-unknown`. The `needle-wasm` crate uses `wasm-bindgen` to expose
the engine to JavaScript. Key constraints vs. native:

- No threads (WASM threads require `SharedArrayBuffer` and COOP/COEP headers)
- No file I/O — weights and vocab must be passed as `Uint8Array` / `String` from JS
- `wasm-opt = false` in Cargo.toml (disables wasm-pack's bundled optimizer); CI runs `wasm-opt -Oz` via system binaryen instead
- SIMD: `matvec_scalar` fallback used for wasm32 (WASM SIMD128 path not yet implemented)

Build command:
```bash
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/
```

---

## Parity testing

The parity test suite compares Rust output to Python/JAX reference outputs at two levels:

1. **E2E token IDs** (`tests/e2e_parity.rs`): exact token ID sequence match for 10 query+tools examples. Reference vectors in `tests/e2e_vectors.json` generated by `tools/gen_e2e_vectors.py`.

2. **Numeric tensors** (`tests/real_parity.rs`): raw encoder hidden states and per-step decoder logits match (within floating-point tolerance) for 5 decode steps. Reference vectors in `tests/real_vectors.json`.

A parity test failure always indicates a real divergence — not numerical noise — because we compare argmax results, not raw logits.

---

## Weight format

SafeTensors file with `__metadata__` containing all model config as JSON strings.

| Tensor | Dtype | Notes |
|---|---|---|
| `embedding` | BF16 | 8192 × 512; also used as tied LM head |
| `encoder.{i}.self_attn.wq` | I4 + `.scale` F32 | packed nibbles, group_size=32 |
| `encoder.{i}.norm` | F32 | ZCRMSNorm γ vector |
| `encoder.{i}.self_attn_gate` | F32 scalar | gated residual gate |
| `decoder.{i}.*` | same pattern | self-attn + cross-attn + optional FFN |
| `encoder_final_norm` / `decoder_final_norm` | F32 | post-stack norm |
| `contrastive_hidden_kernel/bias` | F32 | optional |
| `contrastive_proj_kernel` | F32 | presence triggers contrastive head load |
