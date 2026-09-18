# Changelog

All notable changes to needle-rs are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.3.0] - 2026-09-18

Adds support for **Needle 3** (`Cactus-Compute/needle3`). v3 lands alongside v1
and v2 rather than replacing them; all three run from one binary, one WASM
module and one Python package.

Upstream changed a great deal between v2 and v3 — asymmetric query/value head
widths, a causal convolution over Q/K/V, hybrid local/global attention, five
Engram sites, and a "HadamardMLP" that no longer contains a Hadamard transform.
None of the weights, the container header or the forward pass are shared with
v2, so v3 is a parallel path.

The engineering record, including what was measured and what was rejected, is
in [docs/v3-port-record.md](docs/v3-port-record.md).

### Added

- **Needle 3 container** — `needle-infer::cact::{CactV3, CactV3Geometry}`. The
  header is 49 u32-sized fields against v2's 30; the tag differs by one, so a
  container states its own generation and dispatch never guesses from the file
  name. Verified field for field against `export.read_export`, including all
  581 directory records.
- **Forward pass** — `needle-core::v3`. Token-exact against the reference:
  9.0e-6 relative on the logits with zero argmax mismatches over 57 positions.
- **KV cache** — incremental decode is *bit-identical* to prefill, not merely
  close. Sized to the session rather than to the windows: 1.2 MB for a
  57-token run where reserving every window would cost 14.4 MB.
- **Batched prefill and threading** — 2.00x together over 57 positions
  (477 ms stepped and serial, 238 ms batched and threaded), bit-identical.
  Batching alone is 1.19x; the rest is threading, which helps prefill and not
  decode.
- **Confidence head** — matches the reference to 2e-6.
- **Constrained decoding** over the v3 token table, engaged only inside
  `<tool_call>`.
- **Bindings** — `V3Engine` (Rust, Python), `NeedleV3Wasm` (WASM),
  `needle_v3_*` (C ABI, 12 entry points).
- **`kv_bytes(seq_len)`** on the WASM, C and Python surfaces, because on those
  the caller usually owns the memory budget. The C and Python forms take a
  precision flag and WASM has `kv_bytes_int8`, so the figure matches the cache
  the caller is actually about to allocate.
- **int8 key/value cache** — `kv_precision` on `V3Options`, `--kv-int8` on the
  CLI, and a `kv_int8` flag on the WASM, C and Python `generate`. This is the
  width the container declares in `kv_bits`, quantised exactly as upstream's
  `a8_fake_quant_kv` does, per head, query included. Full context drops from
  42.3 MB of cache to 11.5 MB; a 512-token session from 8.8 MB to 2.3 MB. All
  three test queries produce byte-identical tool calls, and batched prefill
  stays bit-identical to stepped decode under it. Opt-in: `F32` remains the
  default because it is the path verified against the reference.
- **`kv_bytes_at(seq_len, precision)`**, and `kv_bytes_int8` on WASM, so a
  caller sizing a budget gets the figure for the cache it will actually
  allocate. Asserted against the real allocation rather than derived on paper.

### Changed

- The CLI dispatches on the container tag rather than the `.cact` extension,
  since both v2 and v3 use it.
- The chat prompt moved to `needle-infer::prompt`, shared by v2 and v3 —
  upstream assembles it identically, and one copy of the JSON compaction
  matters because pretty-printed schemas change the model's answer.
- `silu`, `rms_unit`, `rms_unit_to` and `sinkhorn` moved to a shared layer.
  v2 keeps its import paths through re-exports.
- The WASM module is **537 KB** with three engines (195 KB over the wire),
  against 413 KB with two. That is the cost of carrying v1, v2 and v3 in one
  module.

### Notes

- **Needle 3 is Apache-2.0** where v1 and v2 were MIT. needle-rs itself remains
  MIT; the licence applies to the weights.
- **v3 exports no contrastive head**, so `retrieve_tools` and
  `encode_contrastive` have no v3 equivalent. The methods are absent rather
  than present-and-always-empty.
- **v3 reasons before answering**, on essentially every query. v2 does so only
  sometimes — 2 of 3 sample prompts on the shipped checkpoint — so a caller that
  treated a `<think>` block as a v3 marker would misclassify v2 output. The
  default token budget is 256 rather than v2's 128, and the bindings expose
  `reasoning()` on both.
- **The confidence head is easier to misuse on v3 than on v2.** It scores a
  completion, not a query. On v2 a bare query scored near zero, so the mistake
  announced itself; on v3 it scores 0.80 against 0.93 for a correct completion
  — a plausible number that means nothing.


## [0.2.1] - 2026-08-20

Documentation only. No runtime, API or weight-format change; `0.2.0` and `0.2.1`
are byte-identical in behaviour. Published so the corrected npm registry page
reaches users, since npm cannot replace the README of a version already out.

### Fixed

- The published WASM module was described as "414 KB", which is wrong in every
  unit — it is 423,330 bytes, i.e. 413 KiB or 423 kB. Corrected everywhere.
- The browser demo reported 156 KB against docs claiming 414 KB, which read as a
  contradiction. Both were measuring real things: the demo reads
  `encodedBodySize`, and Cloudflare serves the module with `content-encoding: br`
  at 159,429 bytes. The published figure is now "413 KB module, 156 KB over the
  wire", matching what the demo shows, and the demo's stat card says so.
- Dropped the "132 KB brotli" claim, which was local `brotli -q 11` that no CDN
  delivers.
- The npm README claimed Needle v1 supports tool retrieval the same way v2 does.
  It does not: the published v1 weights carry no contrastive head, so
  `contrastive_dim()` is `0`, `encode_contrastive()` returns `undefined` and
  `retrieve_tools()` returns `[]`. v2's head is 128-dimensional.
- The documented `retrieve_tools` scores were wrong — the third pair read
  `[1,0.31]` where the measured value is `0.5244`.
- `SHA256SUMS` listed every asset with a `dist/` path prefix, so `sha256sum -c`
  failed for anyone who downloaded the release assets flat. Now bare filenames.


## [0.2.0] - 2026-08-20

Adds support for **Needle v2** (`Cactus-Compute/needle2`). Upstream replaced the
v1 encoder-decoder SAN with a decoder-only architecture in a new container; the
two share no weights, loader, or quantisation scheme, so v2 lands alongside v1
rather than replacing it. Both are supported.

Rationale, measurements and the optimisations that were rejected are in
[BENCHMARKS.md](BENCHMARKS.md); the port's engineering record, including what
changed upstream, is in [docs/v2-port-record.md](docs/v2-port-record.md).

### Added

**Needle v2 runtime**

- `needle-core::v2` — decoder-only forward pass: mHC lane routing with Sinkhorn
  mixing, Engram n-gram memory, HadamardMLP, GQA attention with q/k ZCRMSNorm and
  a sigmoid output gate, sliding-window KV cache, tied LM head.
- `needle-core::cq` — Cactus-Quants weights (2/3/4-bit and ternary, Hadamard-rotated
  Lloyd-Max codebooks). The rotation is applied to the activation rather than the
  weights, so weights are never reconstructed during inference.
- `needle-core::hadamard` — fast Walsh-Hadamard transform, replacing both dense
  Hadamard matmuls.
- `needle-core::v2::batch` — batched, chunked prefill.
- `needle-core::v2::heads` — contrastive and confidence probe heads, pooled with a
  streaming softmax in `O(probes x d_model)` instead of retaining every cell.
- `confidence_for(query, tools, completion)` on all four surfaces (Rust, C ABI,
  WASM, Python). The confidence head scores a completed judgement, so it needs
  the prompt plus the model's own output; passing a bare query to the raw
  `confidence` primitive reads near zero however answerable the query is.
- `needle-infer::cact` — `.cact` container reader. The tensor directory is
  nameless, so the canon is derived from the header geometry and every slot's
  shape validated against it.
- `needle-infer::sp_tokenizer` — SentencePiece BPE decoded from the container's
  embedded piece table. No vocabulary file.
- `needle-infer::v2_engine` — prompt template, greedy and temperature sampling,
  streaming, grammar-constrained decoding, `<tool_call>` / `<think>` extraction.
- Bindings: `needle_v2_*` (C ABI), `NeedleV2Wasm` (WASM), `needle_rs.V2Engine`
  (Python).
- `needle-rs` CLI dispatches on the model file; `--constrain`, `--max-tokens`,
  `--temperature`, `--seed`, `--system`, `--json`, `--prefill-chunk`.

**Other**

- `parallel` feature (rayon) — splits large matvec/matmul row ranges and the
  batched-prefill per-position loops. Enabled for the CLI, C ABI and Python wheel;
  off for WASM and `no_std`. Row splitting leaves each row's arithmetic untouched,
  so results are bit-identical.
- AVX2 runtime dispatch for the packed dot product, selected by CPUID.
- Crate metadata for publishing: keywords, categories, `rust-version = "1.87"`.

### Changed

- **The CLI crate publishes as `needle-rs-cli`**, not `needle-cli`: that name on
  crates.io belongs to an unrelated project, as does `needle-rs`. The installed
  binary is unchanged (`needle-rs`), and `cargo install needle-rs-cli` is the
  install line. `cargo install needle-cli` fetches someone else's tool.
- `needle-wasm` and `needle-python` set `publish = false`. They are distributed
  as the npm and PyPI packages named `needle-rs`; neither was ever a crates.io
  artifact, and this makes that explicit rather than incidental.
- The release workflow now fails before uploading anything if the git tag does
  not match `Cargo.toml` and `pyproject.toml`. npm took its version from the tag
  while crates.io and PyPI took theirs from the manifests, so a tag that outran
  a version bump would have published three different versions, none of them
  retractable.
- The npm package ships a README written for it, with absolute links, plus the
  licence. Previously `files` omitted the README entirely, so the npmjs.com page
  would have rendered from the description alone; wasm-pack's copy of the
  workspace README also carries repo-relative image links that 404 off GitHub.
- KV cache is a ring sized to the attention window rather than `max_seq_len`:
  **14.2 MB instead of 113.3 MB**. `V2Model::make_state_full_causal` allocates the
  full length for the probe heads, which attend over the whole sequence.
- Packed dot product accumulates in 8 independent lanes rather than one running
  scalar, which was latency-bound: **2.9x** on the 2-bit projections, **2.8x** on
  the tied LM head.
- Prefill **5.74 -> 1.60 ms/token** through batching, threading, batched Engram
  projections, and threading the per-position loops.
- `needle-python` uses pyo3 0.29 with `abi3-py38`: one wheel for every CPython
  >= 3.8.

### Fixed

- `needle-python` could not build against any Python newer than 3.12, so
  `cargo build --workspace` failed outright on a current system.
- An aarch64-only `unreachable_statement` warning failed `-D warnings` on exactly
  the ARM targets this crate targets.
- Nothing verified the NEON INT4 kernel against the scalar reference; added a
  differential test.
- The v1 fixture generators broke silently when upstream restructured; each now
  pins the v1 reference commit (`1807b1d`).
- Constrained decoding could repeat an argument key while another declared key was
  still unused (`with_unique_arg_keys`, v2 only, so v1 token parity is unchanged).
- `needle-infer` no longer ships a 511 KB parity fixture no consumer can use.

### Verified

Against `Cactus-Compute/needle2` (13,737,807 bytes, 405 tensors):

- **Forward pass** — 788 capture points across 27 layers match upstream's
  `decode.forward_cached` on the same weights; largest relative deviation 1.9e-5.
- **End to end** — exact generated token ids over 14 prompt/tool combinations
  (2,482 tokens, longest prompt 364, past the 256-token window).
- **Container** — header, directory and codebook match `export.read_export()`
  field for field; all 145 CQ and 259 FP16 tensors match; the canon covers all 405
  exactly once.
- **Tokenizer** — exact token-ID match on 44 cases against both upstream's
  `RefTokenizer` and real `sentencepiece`.
- **Probe heads** — contrastive within 1.4e-6, confidence within 7.2e-5.
- **Batched prefill** — bit-identical to the sequential path across prompt lengths
  1..129 x chunk sizes 1/8/32/128.

Two optimisations were measured and rejected: hand-written NEON for the packed dot
product was *slower* than LLVM's autovectorisation of the same loop (7.1 against
8.1 Gelem/s), and threading decode's small matvecs made decode **3.5x slower**
(135 to 476 ms). Two write-ordering bugs introduced by the KV ring — batched
write-ahead clobbering entries earlier positions still needed, in attention and in
the Engram value ring — were caught by the end-to-end suite before release.

### Docs and examples

- Every example runs on **both** model versions: the browser demo has a model
  picker and adapts its controls to what the loaded version supports, the CLI
  quickstart takes `--v1` / `--v2` / `--both`, the C-ABI example infers the
  version from the file extension, and the DOM editor accepts `?model=v1`.
- `crates/needle-wasm/tests/node_e2e_v2.js` asserts the contract the browser
  demo depends on, across both versions, and runs in CI.
- README rewritten around v2 with v1 documented alongside; `assets/banner.svg`
  and a new `assets/terminal.svg` carry measured numbers.

### Fixed — docs and examples

- `V2Engine::build_prompt` now compacts the tools JSON. The model was trained on
  compact schemas, and the indentation `JSON.stringify(x, null, 2)` produces was
  enough to change the answer: the same query and schema yielded a correct call
  compact and `[]` pretty-printed. Already-compact input is byte-identical, so
  parity is unmoved.
- The C-ABI example freed a pointer ctypes had already converted to a Python
  `bytes`, aborting the process. String returns are now declared `c_void_p`.
- The DOM editor's `tsconfig.json` was missing `"strict": true`, so TypeScript
  did not narrow discriminated unions and `npm run build` failed on correct code.
- The browser demo now settles on the returned text when streaming ends: v1
  strips the `<tool_call>` marker and restores the caller's tool-name casing, so
  the streamed tokens are a progress view rather than the answer.

### Known limitations

- Prefill is slower than the fused dense-f32 reference (1.60 against 0.51
  ms/token). This is the cost of running from 13.7 MB of 2-bit weights rather than
  ~180 MB dequantised, not an implementation gap.
- A tool call can repeat an argument key once every declared key has been used;
  forcing the object closed needs the grammar extended to the value boundary.
- The probe heads need the full-length cache (113 MB). `new_head_state` plus
  `encode_contrastive_with_state` reuses one across calls; `retrieve_tools` does.
- AVX2 correctness is asserted on CI's x86_64 runners; no throughput figure is
  claimed, as development is on aarch64.
- The v2 MTP head is not ported: `export.py` writes no MTP tensors, so the
  container carries none.
- Confidence is the calibration head only. Upstream's published score is the
  minimum of that head and the decode probability of the call tokens; the
  composition lives in Cactus's compiled engine, so there is no reference to
  verify a port against. `confidence_for` exposes the head with the
  prompt-plus-completion input it was trained on.

## [0.1.0] - 2026-05-17

Initial public release.

### Added

- `needle-core`: `no_std` compute kernels — INT4 quantization, ZCRMSNorm, RoPE,
  GQA attention, gated residuals, FFN (SwiGLU/DReLU/GeGLU), tied-embedding LM head
- `needle-infer`: inference engine with SafeTensors loader, BPE tokenizer,
  constrained decoder (JSON Schema + flat format), contrastive retrieval head
- `needle-c`: stable C ABI (`needle_load`, `needle_load_bytes`, `needle_run`,
  `needle_run_stream`, `needle_encode_contrastive`, `needle_contrastive_dim`,
  `needle_retrieve_tools`, `needle_free`, `needle_free_str`, `needle_last_error`)
- `needle-wasm`: WASM bindings via wasm-bindgen (`load`, `run`, `run_stream`,
  `run_batch`, `encode_contrastive`, `retrieve_tools`)
- `needle-cli`: CLI binary with `--stream` flag for per-token output
- AVX2 runtime dispatch (CPUID, no `target-cpu=native`) and NEON (aarch64 baseline)
- 91 tests: 47 Rust unit/integration + 44 Node.js WASM assertions
- E2E parity suite: exact token-ID match against Python/JAX reference (10 examples)
- Constrained decoder handles both flat and JSON Schema tool definitions;
  fixes a latent bug in the Python reference for OpenAI-compatible tool formats

### Performance (Intel i7-1185G7)

- End-to-end load + infer: 283 ms vs ~9,100 ms Python cold-start
- INT4 matvec 512×512 (AVX2): 83 µs / 3.2 Gelem/s
- CLI binary: 533 KB stripped; WASM module: 260 KB (`wasm-opt -Oz`)

[Unreleased]: https://github.com/geekgineer/needle-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/geekgineer/needle-rs/releases/tag/v0.3.0
[0.2.1]: https://github.com/geekgineer/needle-rs/releases/tag/v0.2.1
[0.2.0]: https://github.com/geekgineer/needle-rs/releases/tag/v0.2.0
[0.1.0]: https://github.com/geekgineer/needle-rs/releases/tag/v0.1.0
