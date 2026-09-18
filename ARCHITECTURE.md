# Architecture

This document describes the needle-rs inference runtime. For model architecture
details (training, hyperparameters, research decisions), see the
[Needle repository by Cactus Compute](https://github.com/cactus-compute/needle).

---

## Crate layout

```
crates/
  needle-core/    no_std compute kernels (v1, v2 and v3)
  needle-infer/   std inference engine: loaders, tokenizers, engines
  needle-c/       C ABI cdylib + staticlib
  needle-wasm/    WASM bindings (wasm-bindgen)
  needle-cli/     CLI binary
  needle-python/  PyO3 abi3 extension module
```

The dependency graph is a strict DAG: `needle-core` ← `needle-infer` ← `{needle-c, needle-wasm, needle-cli, needle-python}`.

Directory names and published names differ where crates.io required it:

| Directory | Published as | Registry |
|---|---|---|
| `needle-core`, `needle-infer`, `needle-c` | same name | crates.io |
| `needle-cli` | `needle-rs-cli` (binary: `needle-rs`) | crates.io |
| `needle-wasm` | `needle-rs` | npm |
| `needle-python` | `needle-rs` (module `needle_rs`) | PyPI |

`needle-cli` is renamed because the crates.io names `needle-cli` and `needle-rs`
both belong to unrelated projects. `needle-wasm` and `needle-python` set
`publish = false`: they ship through npm and PyPI, not crates.io.

Three model generations are implemented side by side. They share the crate
layout, the norm/RoPE/attention primitives, and the constrained decoder, and
share nothing else — different weights, containers, and quantisation schemes:

| | Needle v1 | Needle v2 | Needle v3 |
|---|---|---|---|
| Kernels | `quant.rs` (INT4) | `cq.rs`, `hadamard.rs` | `cq.rs`, `v3/kernels.rs` |
| Model | `model.rs`, `layers.rs` | `v2/model.rs`, `v2/batch.rs`, `v2/heads.rs` | `v3/model.rs`, `v3/attention.rs`, `v3/engram.rs`, `v3/mhc.rs`, `v3/cache.rs`, `v3/heads.rs` |
| Container | `safetensors.rs` + `tokenizer.rs` | `cact.rs` + `sp_tokenizer.rs` | `cact.rs` (`CactV3`) + `sp_tokenizer.rs` |
| Engine | `engine.rs` | `v2_engine.rs` | `v3_engine.rs` |

MSRV is 1.87, set by `usize::is_multiple_of`.

---

## needle-core — `no_std` kernels

`needle-core` has `#![no_std]` with `extern crate alloc`. This makes it portable to
embedded targets that lack a system allocator. It has one required dependency: `libm`
(transcendental functions — exp, sin, cos, sqrt). All other arithmetic is standard Rust.

Optional features:

| Feature | Effect |
|---|---|
| `simd` *(default)* | AVX2/NEON intrinsics, CPUID-gated at runtime |
| `std` | Enables `std`; required by `parallel` |
| `parallel` | Splits large matvec/matmul row ranges across a rayon pool. Off for WASM and `no_std`. Row splitting leaves each row's arithmetic untouched, so results are bit-identical to the serial path |
| `bench-internals` | Exposes superseded kernel shapes so a benchmark can measure an optimisation against its own predecessor in one run |

### INT4 quantization — Needle v1 (`quant.rs`)

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

### SAN model — Needle v1 (`model.rs`)

Encoder: 12 layers of `encoder_layer_forward` (self-attention only) → final norm → cross-attn KV projection.

Decoder: 8 layers of `decoder_layer_forward` (self-attn + cross-attn + optional FFN) → final norm → LM head.

Tied embedding: `logits[v] = dot(hidden, embedding[v])`. No separate output projection matrix.

---

### Cactus-Quants — Needle v2 (`cq.rs`)

v2 weights are packed 2, 3 or 4 bits (or ternary) against a shared Lloyd-Max
codebook, in groups of 128 along the reduction axis. Each group carries one FP16
L2 norm. `needle2.cact` uses a mixed scheme — `embedding=4,mhc=4,default=2`,
about 2.2 bits effective.

A group reconstructs as:

```
w_g = (codebook[idx_g] * norm_g) @ H        H = Walsh(group) / sqrt(group)
```

**Weights are never reconstructed.** `H` is symmetric, so for an activation slice:

```
dot(x_g, u_g @ H) == dot(H @ x_g, u_g)      u_g = codebook[idx_g] * norm_g
```

The rotation moves off the weights and onto the activation — paid once per matrix
instead of once per `(row, group)`. At the shipped 512x512 that is 4 transforms
instead of 512, and the inner loop reduces to a dot product against packed bytes
with one FP16 scale per group. `prepare_input` exposes the rotation separately so
a layer's `q/k/v/gate` projections, which all read the same activation, pay it
once between them.

The accumulator is split into eight independent lanes (`ACC_LANES`). A single
running accumulator makes the loop latency-bound — every FMA waits on the
previous — which caps throughput regardless of issue width.

`matmul_rows_prepared` evaluates many positions in one pass over the weights,
decoding each group once for the whole batch. It is **bit-identical** to repeated
`matvec_rows_prepared`: the group norm is applied after the group dot and the dot
is summed in the same lane order, so nothing is reassociated.

On x86_64 the same source is recompiled under `avx2,fma` and selected by CPUID,
because baseline x86_64 guarantees only SSE2 and this crate does not build with
`target-cpu=native`. There is no hand-written SIMD: measured on aarch64, LLVM's
autovectorisation of this loop beats a hand-written NEON version.

### Fast Walsh-Hadamard transform (`hadamard.rs`)

Both places Needle uses a Hadamard matrix — quantisation groups and HadamardMLP —
build it by the same Sylvester recursion scaled by `1/sqrt(n)`. Applying it as a
dense matmul costs `n^2` MACs; the butterfly costs `n log2(n)` add/sub and no
multiplies. At `n = 512` that is 4,608 against 262,144.

### Needle v2 model (`v2/`)

Decoder-only, 27 layers at the shipped geometry. Ported from
`needle/model/decode.py::_forward_cached` — upstream's *incremental* reference —
rather than the training graph, because only the decode form carries a KV cache.

| Module | Contents |
|---|---|
| `v2/config.rs` | geometry from the container header; nothing is a compile-time constant |
| `v2/kernels.rs` | `rms_unit`, Sinkhorn, HadamardMLP, the Engram hash |
| `v2/model.rs` | the forward pass, KV ring, Engram history ring |
| `v2/batch.rs` | batched, chunked prefill |
| `v2/heads.rs` | contrastive and confidence probe heads |

One token is processed per call, which covers prefill and decode alike: with the
KV cache and the Engram history ring in place, a step at position `p` depends only
on state.

**mHC lanes.** The residual stream is four lanes wide. Each layer routes them:
`hpre` mixes lanes down to one vector, the block runs on that, and `hpost` plus a
Sinkhorn-normalised `hres` matrix mixes the result back out. All per-position.

**Engram.** N-gram hash memory at layers 2 and 15. `_engram_kv` builds a window of
the token history and applies `_shift_right` along it; read at the kept position
that collapses to plain history lookups — the hash for an order-`o` table reads
tokens `p .. p-(o-1)`, and the value convolution reads `v(p - j * dilation)` for
each tap. So the only state needed is the token history plus the un-convolved
value vectors for the last `conv_taps * dilation` positions.

**Attention.** GQA with per-head ZCRMSNorm on q and k, RoPE, a sigmoid gate from
`gate_proj` applied to the concatenated heads, then `out_proj`.

**KV cache is a ring.** v2 attends over a 256-token window, so the cache holds
`kv_window` positions rather than `max_seq_len` — 14.2 MB instead of 113.3 MB.
Writing slot `pos % cache_len` overwrites position `pos - cache_len`, exactly the
one that just left the window. Two consequences the tests pin:

- attention must write and attend one position at a time; writing a whole chunk
  first clobbers the oldest key earlier positions still need
- the Engram value ring holds only `conv_taps * dilation + 1` positions, so a
  chunk's values live in per-position scratch until every tap has read them

The probe heads are the exception: they attend over the whole sequence, so they
allocate the full length via `make_state_full_causal`. `set_kv_window` refuses a
window wider than the allocated ring rather than reading stale slots.

**Probe heads.** Contrastive retrieval and confidence pool over *cells* — the
scaled embedding plus the lane-mean of the residual stream after every layer, at
every position. Materialised that is `T x (L+1) x d_model`, 117 MB at full
context. A softmax-weighted average is what an online (running-max) softmax
computes, so cells are streamed straight out of the forward pass and the pool
keeps `O(probes * d_model)` state — 16 KB for the confidence head.

Note the two heads default to `window = 0` (full causal) upstream while the LM
path runs the checkpoint's `kv_window`. Below 256 tokens the two agree bit for
bit; above it they diverge, so this is a real fork rather than a rounding detail.

---

## needle-infer — inference engine

### SafeTensors reader — Needle v1 (`safetensors.rs`)

Self-contained parser: 8-byte LE header length + JSON metadata + raw data blobs.
Handles `F32`, `BF16`, `F16`, `I8`, and a custom `I4` dtype (packed nibbles).
No external parsing dependency.

### BPE tokenizer — Needle v1 (`tokenizer.rs`)

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

### Engine — Needle v1 (`engine.rs`)

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

### `.cact` container reader — Needle v2 (`cact.rs`)

A 120-byte geometry header, the shared codebook, then a **nameless** positional
tensor directory and 64-byte-aligned blobs. Because the directory carries no
names, the canon is derived from the header geometry and every slot's shape is
validated against it — a layout drift fails at load rather than producing wrong
logits.

One container carries the weights, the geometry *and* the tokenizer, so a v2
model is a single file with no vocabulary and no config side-car.

### SentencePiece tokenizer — Needle v2 (`sp_tokenizer.rs`)

Decoded from the container's embedded piece table: no vocabulary file and no new
dependency. A port of upstream's `RefTokenizer`, which is the normative
encoder/decoder for the blob format — and which was verified against real
`sentencepiece` before porting.

### Engine — Needle v2 (`v2_engine.rs`)

Prompt assembly, greedy and temperature sampling, streaming, grammar-constrained
decoding, `<tool_call>` / `<think>` extraction, and the probe heads.

The tools JSON is **compacted** before embedding in the prompt. This is not
cosmetic: the model was trained on compact schemas, and the indentation
`JSON.stringify(x, null, 2)` produces is enough to change the decision — the same
query and schema yields a correct call compact and `[]` pretty-printed.

Constrained decoding reuses the v1 JSON state machine over a v2 token table, and
is engaged only between `<tool_call>` and `</tool_call>`. Both are single tokens,
so entering and leaving is an id comparison; running the machine over the whole
turn would risk a `"name":"` inside `<think>` prose putting it into a constrained
state where it does not belong.

---

## WASM build details

Target: `wasm32-unknown-unknown`. The `needle-wasm` crate uses `wasm-bindgen` to expose
the engine to JavaScript. Key constraints vs. native:

- No threads — the `parallel` feature is off for wasm32, so kernels run serially
- No file I/O; a model is passed in as `Uint8Array` (plus a vocabulary `String` for v1)
- `wasm-opt = false` in Cargo.toml (disables wasm-pack's bundled optimizer); CI runs `wasm-opt -Oz` via system binaryen instead
- SIMD: the portable kernel is used for wasm32 (no SIMD128 path yet)

One module exports `NeedleWasm` (v1), `NeedleV2Wasm` (v2) and `NeedleV3Wasm`
(v3): **537 KB** after `wasm-opt -Oz`, 195 KB over the wire as Cloudflare
serves it (162 KB at `brotli -q 11` if you compress it yourself).

```bash
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/
```

---

## Parity testing

All three engines are compared against the Python/JAX reference. Every fixture
generator lives in `tools/`, and each suite skips with a printed notice when its
inputs are absent, so a fresh clone runs `cargo test` clean.

**Needle v1**

| Suite | Checks |
|---|---|
| `e2e_parity.rs` | exact token-ID sequences over 560 generated examples |
| `real_parity.rs` | encoder hidden states and per-step decoder logits |

**Needle v2** — the reference is run on weights rebuilt from the shipped container
by `tools/cact_params.py`, which inverts `export._tensors`. That matters: the
container's directory is positional, so a *correct* tensor can still land in the
wrong slot and every per-tensor comparison still pass.

| Suite | Checks |
|---|---|
| `v2_forward_parity.rs` | 788 captured intermediates across 27 layers, max relative deviation 1.9e-5 |
| `v2_e2e_parity.rs` | exact token ids over 14 prompt/tool combinations, longest prompt past the window |
| `cact_parity.rs` | header, directory, codebook, all 145 CQ and 259 FP16 tensors |
| `tokenizer_v2_parity.rs` | exact ids on 44 cases vs `RefTokenizer` *and* `sentencepiece` |
| `v2_heads_parity.rs` | contrastive 1.4e-6, confidence 7.2e-5 |
| `v2_batch_parity.rs` | batched prefill bit-identical to sequential |
| `v2_constrained.rs` | payloads confined to the declared schema |
| `node_e2e_v2.js` | the WASM surface, v2 and v1 |
| `v3_forward_parity.rs` | v3 forward 9.0e-6, decode bit-identical to prefill |
| `v3_component_parity.rs` | v3 MLP 5.4e-6, attention 2.7e-6, Engram exact |
| `cact_v3_parity.rs` | the 196-byte header and all 581 directory records |
| `v3_kv_quant_parity.rs` | int8 KV against upstream `fake_quant`, exactly |
| `v3_kv_int8.rs` | the quantised cache saves the memory it claims |
| `v3_statelessness.rs` | nothing survives between calls on a shared engine |
| `node_e2e_v3.js` | the WASM surface, v3 |

`v2_e2e_parity.rs` is the acceptance gate for invasive changes: it caught both
write-ordering bugs introduced by the KV ring before release.

---

## Weight formats

### Needle v2 — `.cact`

A single container. Byte layout is specified by `needle/model/export.py`; the
reader is `needle-infer::cact`.

```
header      120 bytes: 29 u32 geometry fields, then rope_theta as f32
codebook    codebook_len * f32   (cb2 | cb3 | cb4, pre-scaled by 1/sqrt(group))
directory   num_tensors * 44-byte records, NO names — tensors are positional
blobs       64-byte aligned
```

Tensor order is the canon `export.py` documents: embedding; then fourteen tensors
per layer; then the nine mHC blocks; then four per Engram site; then `final_norm`;
then the optional probe heads; then the RAW tokenizer. `needle2.cact` is
13,737,807 bytes and 405 tensors — 141 CQ at 2 bits, 4 at 4 bits, 259 FP16, and
1 RAW.

A CQ blob is the packed indices followed by the per-group FP16 norms. Indices are
one continuous LSB-first bitstream per row; ternary instead stores four signed
2-bit crumbs per byte.

### Needle v1 — SafeTensors

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
