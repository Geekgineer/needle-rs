# Benchmarks

All numbers in this file were measured on the hardware specified below. Raw benchmark
code lives in `crates/needle-core/benches/` and `crates/needle-infer/tests/`.

## Hardware

| Field | Value |
|---|---|
| CPU | Intel Core i7-1185G7 (Tiger Lake, 4 cores / 8 threads, 3.0 GHz base / 4.8 GHz boost) |
| Memory | LPDDR4x, dual-channel |
| OS | Linux (kernel 5.15) |
| Rust | 1.89 stable, `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"` |

Apple Silicon (M-series) numbers are not yet measured. The NEON path is implemented
and verified to compile for aarch64, but `cargo bench` results are pending.
Contributions welcome — see CONTRIBUTING.md.

---

## End-to-End Inference Latency

Full pipeline: load weights from disk + tokenize + encode + decode + post-process.
Measured with the CLI binary (`needle-rs`), median of 5 runs.

| Runtime | Scenario | Latency |
|---|---|---|
| **needle-rs (Rust)** | load + infer | **283 ms** |
| Python / JAX | first infer (includes XLA JIT compile) | 7,229 ms |
| Python / JAX | warm infer (JIT already compiled) | 4,389 ms |
| Python / JAX | cold start (import + load + first infer) | ~9,100 ms |

The Python numbers include: 1,622 ms import (`jax`, `flax`, etc.) + 7,229 ms first run.

---

## INT4 Matrix-Vector Multiply (AVX2, hot kernels)

Every attention projection (Q/K/V/O) and FFN linear layer calls `QuantizedWeight::matvec`.
Measured with `cargo bench -p needle-core -- matvec`.

| Shape (in × out) | Kernel usage | Median | Throughput |
|---|---|---|---|
| 512 × 512 | Q/K/V/O projection (d_model=512) | **83 µs** | 3.2 Gelem/s |
| 512 × 256 | KV projection (4 KV heads × 64) | **41 µs** | 3.3 Gelem/s |
| 2048 × 512 | FFN down-projection | **311 µs** | 3.1 Gelem/s |
| 512 × 2048 | FFN up/gate-projection | **309 µs** | 3.2 Gelem/s |

"Elements" = (input features × output features) processed per second (dequantize + multiply + accumulate).

---

## ZCRMSNorm

| Sequence length | Median |
|---|---|
| 16 tokens | 59 ns |
| 512 tokens | 773 ns |
| 2048 tokens | 3.05 µs |

---

## SIMD Coverage

| Architecture | Kernel | Detection |
|---|---|---|
| x86_64 | `matvec_avx2` (256-bit FMA) | Runtime CPUID via `is_x86_feature_detected!("avx2")` — no `target-cpu=native` required |
| aarch64 | `matvec_neon` (128-bit, 8 output features/step) | Unconditional — NEON is mandatory in ARMv8 |
| wasm32 / any | `matvec_scalar` | Fallback for all other targets |

---

## Binary / Deployment Size

| Artifact | Size | Notes |
|---|---|---|
| CLI binary (`needle-rs`) | **533 KB** | stripped release, `lto="fat"` |
| C shared library (`libneedle_c.so`) | **557 KB** | cdylib, stable C ABI |
| WASM module (`needle_wasm_bg.wasm`) | **260 KB** | `wasm32-unknown-unknown`, `wasm-opt -Oz` applied |
| Weight file (`needle.safetensors`) | **22 MB** | INT4 packed nibbles + BF16 norms |
| Vocabulary (`vocab.txt`) | **120 KB** | 8,192 SentencePiece pieces, text format |
| **Total deploy (binary + weights + vocab)** | **~23 MB** | |

---

## Dependency Count

| Runtime | External deps | Notes |
|---|---|---|
| needle-rs native binary | **1** (`libm`) | `no_std` core; only dep is transcendentals |
| needle-rs WASM | **4** | `serde_json`, `wasm-bindgen`, `js-sys`, `web-sys` (browser glue only) |
| Python/JAX reference | **12** | `jax`, `jaxlib`, `flax`, `optax`, `datasets`, `huggingface_hub`, `gcsfs`, `transformers`, `wandb`, `pyyaml`, `sentencepiece`, `google-genai` |

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
