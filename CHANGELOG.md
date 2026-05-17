# Changelog

All notable changes to needle-rs are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/geekgineer/needle-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/geekgineer/needle-rs/releases/tag/v0.1.0
