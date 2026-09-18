# Benchmarks

All numbers in this file were measured on the hardware specified below. Raw benchmark
code lives in `crates/needle-core/benches/` and `crates/needle-infer/tests/`.

## Hardware

### x86_64 (AVX2)

| Field | Value |
|---|---|
| CPU | Intel Core i7-1185G7 (Tiger Lake, 4 cores / 8 threads, 3.0 GHz base / 4.8 GHz boost) |
| Memory | LPDDR4x, dual-channel |
| OS | Linux (kernel 5.15) |
| Rust | 1.89 stable, `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"` |

### aarch64 (NEON) — v1 numbers

| Field | Value |
|---|---|
| CPU | Apple M4 Max (16-core: 12 P + 4 E) |
| Memory | 64 GB unified |
| OS | macOS 26.0.1 |
| Rust | 1.95.0 stable, `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"` |

### aarch64 (NEON) — v2 and v3 numbers

Every figure in the *Needle v3* and *Needle v2* sections below was measured on
this machine, which is **not** the one above. v1 numbers are not comparable with
either. The v3 figures were taken on macOS 27.0; the v2 figures predate that
upgrade.

| Field | Value |
|---|---|
| CPU | Apple M5 Max (18-core) |
| Memory | 64 GB unified |
| OS | macOS 26.6.2 |
| Rust | 1.97.1 stable, `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"` |

---

## Needle v3 (`.cact`, `Cactus-Compute/needle3`)

Measured on the **M5 Max** entry above (macOS 27.0), built with
`--features parallel` — which is what the CLI, Python and C crates enable, so it
is what a user actually runs. Model: `needle3.cact`, 35,335,380 bytes, d_model
768, 20 layers, 12 query / 2 KV heads with asymmetric head dims (qk 48, v 64),
vocab 8192, context 8192, `sliding_window` 1024 with global attention at layers
4/9/14/19, Engram at five sites.

```
cargo run --release -p needle-infer --features parallel --example v3_bench
```

### Against v2, same machine, same build flags

| Phase | v3 | v2 | |
|---|---|---|---|
| Load model | **10 ms** | 9 ms | |
| Prefill, per token | 4.20 ms | **1.46 ms** | 0.35x |
| Decode, per token | 8.65 ms | **7.05 ms** | 0.82x |

v3 carries 121M parameters against v2's 45M. Being 2–3x slower per token for
2.7x the weights is the trade, and it is the whole reason both generations stay
in the tree: v2 is not deprecated by v3, it is the faster, smaller option.

A 100-token prompt answers in about 700 ms end to end, 424 ms of which is time
to first token.

### Threading helps v3's prefill, not its decode

| Phase | serial | `--features parallel` | |
|---|---|---|---|
| Prefill, per token | 7.61 ms | **4.20 ms** | 1.81x |
| Decode, per token | 9.48 ms | **8.65 ms** | 1.10x |

The same finding v2 reached, more pronounced: prefill has whole matrices to
split, while a decode step is a single-position matvec too small to pay the
dispatch. Batching prefill rather than stepping it is worth a further 1.19x
(`v3_forward_parity` prints it), and is bit-identical.

### The key/value cache

v3's cache is large enough to be a deployment decision rather than a footnote —
4 global layers hold the full context while 16 local layers hold a 1024-token
window.

| Session | f32 | int8 |
|---|---|---|
| 512 tokens | 8.8 MB | **2.3 MB** |
| 2048 tokens | 21.0 MB | **5.6 MB** |
| 8192 tokens (full context) | 42.0 MB | **11.2 MB** |

int8 is the width the container declares in `kv_bits`, quantised exactly as
upstream's `a8_fake_quant_kv` does. It is 27% of f32 rather than 25% because
each stored head vector carries its own `f32` scale, and it costs about 2% of
speed — prefill 4.29 against 4.20 ms/token, decode 8.72 against 8.65 — so it is
very close to a free 3.7x on the dominant memory cost. Opt-in via `--kv-int8`;
see [docs/v3-port-record.md](docs/v3-port-record.md) for what it does and does
not preserve.

---

## Needle v2 (`.cact`, `Cactus-Compute/needle2`)

Measured on the **M5 Max** entry above. Model: `needle2.cact`, 13,737,807 bytes,
d_model 512, 27 layers, 8/4 heads, vocab 8192, kv_window 256, Engram sites (2, 15),
mixed Cactus-Quants `embedding=4,mhc=4,default=2`.

```
cargo run   --release -p needle-infer --example v2_bench          # end to end
cargo bench -p needle-core --features bench-internals \
            --bench cq_kernels                                    # kernels
```

### On reading these numbers

**Every absolute figure here is steady state, under sustained load.** This
machine downclocks about 1.5x once the benchmarks have been running for a while:
the same 512x512 kernel measures 12.2 Gelem/s on a cold clock and 8.1 Gelem/s
warm. Sustained inference sees the warm figure, so that is what is quoted.

The consequence for *ratios* is that two numbers recorded minutes apart are not
comparable. Every ratio below was measured inside a single benchmark run against
its own predecessor, which is what the `bench-internals` feature exists for — it
exposes the earlier kernel shapes so they can be timed side by side rather than
from memory.

### End to end, and where the crossover is

Prefill and decode stand in opposite directions against the reference, so a
single query time says little on its own. Reference figures are from
`tools/cact_params.py` feeding `decode.forward_cached` — same machine, same
weights, JIT already warm.

| Phase | needle-rs | Python / JAX | |
|---|---|---|---|
| Load model | **9 ms** | 761 ms (params) + 26.5 s first-call JIT | **~50x** |
| Prefill, per token | 1.60 ms | **0.51 ms** | 0.32x |
| Decode, per token | **8.2 ms** | 11.0 ms | **1.35x** |

So decode is faster here and prefill is slower, which makes the answer depend on
how much is generated. `examples/v2_crossover` measures both and reports the
crossing:

| Tokens generated | needle-rs | JAX | |
|---|---|---|---|
| 8 | 253 ms | 141 ms | 0.56x |
| 19 | 343 ms | 262 ms | 0.76x |
| 40 | 514 ms | 493 ms | 0.96x |
| **64** | **710 ms** | 757 ms | **1.07x** |
| 128 | 1,233 ms | 1,461 ms | 1.18x |
| 512 | 4,368 ms | 5,685 ms | 1.30x |

**needle-rs is faster from about 48 generated tokens upward**, and the margin
grows with length. Below that the reference's prefill advantage dominates. Note
that a tool call with a reasoning trace routinely exceeds 48 tokens — 8 of the 14
cases in `tests/v2_e2e_vectors.json` do — while a bare call without reasoning does
not.

Two things the table does not show. The reference's numbers are *warm*: reaching
them costs 26.5 s of XLA compilation and 761 ms of parameter loading, against 9 ms
here, so on a cold process needle-rs wins at any length. And it runs from 13.7 MB
of packed weights against ~180 MB dequantised to f32, which is the same trade that
costs the prefill.

### Why prefill is the slower half

The reference evaluates prefill as fused dense f32 GEMM. This runtime evaluates it
as quantised-decode GEMM over 2-bit weights: the packed kernel reaches 38 Gelem/s
threaded, but a BLAS-class f32 GEMM on already-resident weights is simply faster
per multiply-add. That is the cost side of using 13x less memory, not an
implementation gap that a better kernel closes.

What *was* an implementation gap has been removed. Prefill went from 5.74 ms/token
to 1.60 — **3.6x** — in four steps, each exact:

| | ms/token |
|---|---|
| One position at a time, serial | 5.74 |
| Batched projections (chunk 128) | 4.15 |
| Row-split threading on the batched matmuls | 2.05 |
| Batched Engram projections | 2.03 |
| Per-position loops threaded (MLP, mHC, norms, rotations) | **1.60** |

Decode is deliberately *not* threaded. Its 135 matvecs per token are ~30 µs each,
so an eight-way split leaves ~4 µs per task against a rayon dispatch of the same
order: measured, that makes decode **3.5x slower** (135 ms to 476 ms for 19
tokens). Only the tied LM head clears the bar. Threading is enabled by the
`parallel` feature — on for the CLI, C ABI and Python wheel, off for WASM and
`no_std`.

### Cactus-Quants kernels

One element is one multiply-accumulate against a packed weight; weights are never
reconstructed.

| Kernel | Shape | Width | Time | Throughput |
|---|---|---|---|---|
| `q_proj` / `gate_proj` / `out_proj` | 512x512 | 2-bit | 32.2 µs | 8.1 Gelem/s |
| `k_proj` / `v_proj` | 256x512 | 2-bit | 16.7 µs | 8.1 Gelem/s |
| tied LM head | 8192x512 | 4-bit | 700 µs | 6.0 Gelem/s |

Supporting kernels: `prepare_input` 1.06 µs at 512 and 4.22 µs at 2048 (mHC);
`fwht` 0.28 µs at 128 and 1.10 µs at 512.

### Where the kernel speed came from

All three variants timed in one run, so the clock is common. "Reference" is the
obvious implementation — read each index out of the bitstream, one running
accumulator. "Single accumulator" adds the byte LUT but keeps one accumulator.
"Optimised" splits the accumulator into 8 independent lanes.

| | 512x512 @ 2-bit | 8192x512 @ 4-bit |
|---|---|---|
| Reference | 0.89 Gelem/s | 0.89 Gelem/s |
| Byte LUT, single accumulator | 2.80 | 2.10 |
| Byte LUT, 8 lanes (shipped) | **8.15** | **5.96** |
| Lane split alone | **2.91x** | **2.83x** |
| Reference to shipped | **9.1x** | **6.7x** |

A single running accumulator makes the loop latency-bound: every FMA waits on the
previous one, so throughput is capped at one element per FMA latency however wide
the machine is. Independent lanes expose parallelism the hardware already had.

Four things were tried and rejected, all measured:

| Attempt | Result |
|---|---|
| 16 accumulator lanes | 8.87 Gelem/s — worse, register pressure |
| 32 accumulator lanes | 7.95 Gelem/s — worse still |
| FMA straight from the LUT, no staging array | 6.68 Gelem/s — 18% worse; staging is what lets the autovectoriser emit whole-vector loads |
| Hand-written NEON, 4 vector accumulators | ~7.1 Gelem/s — **slower than LLVM's autovectorisation of the same loop** |

The last one is why this crate ships no hand-written SIMD for the packed dot
product. On x86_64 the same source is instead recompiled under `avx2,fma` and
selected by CPUID, because baseline x86_64 guarantees only SSE2 and this crate
deliberately avoids `target-cpu=native` — the win there is *dispatch*, not
intrinsics. That path is exercised by CI on x86_64, not measured here.

### Memory

Reported by `cargo run --release -p needle-infer --example v2_sizes`, not
estimated:

| | Size |
|---|---|
| Model file | 13.74 MB |
| KV cache, generation (ring of 256 = the attention window) | **14.16 MB** |
| KV cache, full causal (2048 positions; probe heads only) | 113.25 MB |
| Batched-prefill scratch, chunk 128 (default) | 8.41 MB |
| Contrastive pool (4 probes, streaming) | 8.2 KB |
| Confidence pool (8 probes, streaming) | 16.4 KB |
| Cells the reference would retain instead of streaming | 117.4 MB |

A generation session is therefore about **23 MB** of working memory on top of the
13.7 MB model, against 113 MB before the KV cache became a ring — `kv_window` is
256, so the LM path never reads further back than that and a ring of that width is
sufficient. Writing slot `pos % 256` overwrites position `pos - 256`, exactly the
one that has just left the window.

The probe heads are the exception, and they are now the outlier rather than the
norm: `encode_contrastive` and `forward_confidence` run full causal, so they need
the full-length cache — **113 MB, five times a generation session**.
`V2State::set_kv_window` refuses a window larger than the allocated ring rather
than silently reading stale slots, so this cannot be papered over.

A single-shot `encode_contrastive` therefore allocates and frees 113 MB. Use
`new_head_state` plus `encode_contrastive_with_state` to hold one across calls;
`retrieve_tools` does exactly that, so ranking N descriptions allocates once
rather than N+1 times.

The ring also made two write-ordering bugs possible, both caught by
`tests/v2_batch_parity.rs` and both fixed by interleaving rather than batching the
writes: attention must write and attend one position at a time (writing a whole
chunk first clobbers the oldest key earlier positions still need), and the Engram
value ring holds only 13 positions, so a chunk's values live in per-position
scratch until every tap convolution has read them.

### Algorithmic notes

Operation counts, not measurements:

- A Cactus-Quants group reconstructs as `w_g = u_g @ H` with `H` symmetric, so
  `dot(x_g, w_g) == dot(H @ x_g, u_g)`. Rotating the *activation* costs one
  transform per group for the whole matrix; rotating the *weights* would cost one
  per `(row, group)`. At 512x512 that is 4 transforms instead of 512.
- `H` is applied as a fast Walsh-Hadamard transform: `n log2(n)` add/sub against
  `n^2` multiply-accumulates — 4,608 against 262,144 at `hada_n = 512`, twice per
  layer inside HadamardMLP.
- The probe heads pool over `T * (L + 1)` cells. Computed the reference's way that
  is 117 MB of retained activations at `T = 2048`; an online softmax reaches the
  same result in `O(probes * d_model)` — 16 KB for the confidence head.

## Needle v1 (`.safetensors`)

## End-to-End Inference Latency

Full pipeline: load weights from disk + tokenize + encode + decode + post-process.
Measured with the CLI binary (`needle-rs`), median of 5 runs.

| Runtime | Scenario | Latency |
|---|---|---|
| **needle-rs (Rust, AVX2 — i7-1185G7)** | load + infer | **283 ms** |
| **needle-rs (Rust, NEON — M4 Max)** | load + infer | **~100 ms** |
| Python / JAX | first infer (includes XLA JIT compile) | 7,229 ms |
| Python / JAX | warm infer (JIT already compiled) | 4,389 ms |
| Python / JAX | cold start (import + load + first infer) | ~9,100 ms |

The Python numbers include: 1,622 ms import (`jax`, `flax`, etc.) + 7,229 ms first run.

---

## INT4 Matrix-Vector Multiply (hot kernels)

Every attention projection (Q/K/V/O) and FFN linear layer calls `QuantizedWeight::matvec`.
Measured with `cargo bench -p needle-core -- matvec`.

### AVX2 (Intel i7-1185G7)

| Shape (in × out) | Kernel usage | Median | Throughput |
|---|---|---|---|
| 512 × 512 | Q/K/V/O projection (d_model=512) | **83 µs** | 3.2 Gelem/s |
| 512 × 256 | KV projection (4 KV heads × 64) | **41 µs** | 3.3 Gelem/s |
| 2048 × 512 | FFN down-projection | **311 µs** | 3.1 Gelem/s |
| 512 × 2048 | FFN up/gate-projection | **309 µs** | 3.2 Gelem/s |

### NEON (Apple M4 Max)

| Shape (in × out) | Kernel usage | Median | Throughput |
|---|---|---|---|
| 512 × 512 | Q/K/V/O projection (d_model=512) | **28.76 µs** | 9.11 Gelem/s |
| 512 × 256 | KV projection (4 KV heads × 64) | **14.33 µs** | 9.14 Gelem/s |
| 2048 × 512 | FFN down-projection | **115.66 µs** | 9.07 Gelem/s |
| 512 × 2048 | FFN up/gate-projection | **113.79 µs** | 9.21 Gelem/s |

"Elements" = (input features × output features) processed per second (dequantize + multiply + accumulate).

---

## ZCRMSNorm

| Sequence length | AVX2 (i7-1185G7) | NEON (M4 Max) |
|---|---|---|
| 16 tokens | 59 ns | **28.0 ns** |
| 512 tokens | 773 ns | **290.8 ns** |
| 2048 tokens | 3.05 µs | **1.13 µs** |

---

## SIMD Coverage

| Architecture | Kernel | Detection |
|---|---|---|
| x86_64 | `matvec_avx2` (256-bit FMA) | Runtime CPUID via `is_x86_feature_detected!("avx2")` — no `target-cpu=native` required |
| aarch64 | `matvec_neon` (128-bit, 8 output features/step) | Unconditional — NEON is mandatory in ARMv8 |
| wasm32 / any | `matvec_scalar` | Fallback for all other targets |

---

## Binary / Deployment Size

One binary carries all three engines: there is no v1-only, v2-only or v3-only
build. Code sizes below are macOS/aarch64 (`lto="fat"`); a Linux/x86_64 build of
the same tree comes out smaller, so treat these as the upper bound.

Re-measured for 0.3.0 on the M5 Max entry, with all three generations compiled
in. Every figure grew when v3 landed; the earlier numbers in this table were for
a two-generation build on an M4 Max and are not comparable.

| Artifact | Size | Notes |
|---|---|---|
| CLI binary (`needle-rs`) | **765 KB** | stripped release, all three engines |
| C shared library (`libneedle_c.dylib`) | **815 KB** | cdylib, stable C ABI, all three surfaces |
| WASM module (`needle_wasm_bg.wasm`) | **537 KB** | after `wasm-opt -Oz`; **195 KB** as Cloudflare Pages actually serves it (brotli), 204 KB gzipped, 162 KB if you compress it yourself at `brotli -q 11` |
| — same module, unoptimised | 605 KB | what `wasm-pack build` alone emits |

`wasm-opt` is **not** run by `wasm-pack` here — the crate sets
`wasm-opt = false`, because the binary wasm-pack downloads fails in this build
environment. Run it yourself for the 537 KB figure:

```bash
wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
  pkg/needle_wasm_bg.wasm -o pkg/needle_wasm_bg.wasm
```

Weights, per version:

| Version | Files | Size |
|---|---|---|
| v3 | `needle3.cact` (weights + geometry + tokenizer) | **35.3 MB** |
| v2 | `needle2.cact` (weights + geometry + tokenizer) | **13.7 MB** |
| v1 | `needle.safetensors` + `vocab.txt` | **22 MB** + 122 KB |

Smallest complete browser deployment is still v2: 537 KB of runtime plus a
13.7 MB container, 195 KB + 13.7 MB over the wire as served. A generation
session needs roughly 23 MB of working memory on top (see [Memory](#memory)).
v3 is the same runtime with a 35.3 MB container and a larger cache — see the
Needle v3 section.

---

## Dependency Count

| Runtime | External deps | Notes |
|---|---|---|
| needle-rs native binary | **1** (`libm`) | `no_std` core; only dep is transcendentals |
| needle-rs WASM | **4** | `serde_json`, `wasm-bindgen`, `js-sys`, `web-sys` (browser glue only) |
| Python/JAX reference | **12** | `jax`, `jaxlib`, `flax`, `optax`, `datasets`, `huggingface_hub`, `gcsfs`, `transformers`, `wandb`, `pyyaml`, `sentencepiece`, `google-genai` |

On disk: the CPU-only virtualenv used to generate the parity fixtures here
(`jax`, `flax`, `numpy` and their transitive deps, Python 3.12) measures
**479 MB**, of which `jaxlib` alone is 268 MB. A CUDA build is several times
that. The equivalent needle-rs deployment is a 765 KB binary, or 537 KB of
WebAssembly.

---

## Reproducing

```bash
# Install Rust stable
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/geekgineer/needle-rs
cd needle-rust

# Export weights (requires Python + JAX environment and needle.pkl checkpoint)
PYTHONPATH=needle python tools/export.py --checkpoint needle/checkpoints/needle.pkl

# End-to-end latency (CLI, 5 runs)
time for i in 1 2 3 4 5; do
  ./target/release/needle-rs weights/needle.safetensors weights/vocab.txt \
    "What is the weather in Paris?" \
    '[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
done

# Microbenchmarks (matvec, norm)
cargo bench -p needle-core
```
