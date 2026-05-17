# Contributing to needle-rs

Thank you for your interest in contributing. needle-rs is a Rust + WASM runtime for the [Needle](https://github.com/cactus-compute/needle) tool-calling model by Cactus Compute.

## Before you start

- The model architecture, training, and weights are Cactus Compute's work, not ours. Contributions to this repo are about the **deployment runtime**: Rust inference, WASM packaging, C ABI, CLI, quantization, and testing.
- All contributions must preserve exact numerical parity with the Python reference. If your change affects inference output, add or update parity tests in `crates/needle-infer/tests/`.

## How to contribute

1. **Open an issue first** for anything non-trivial. Use [GitHub Issues](https://github.com/geekgineer/needle-rs/issues) to describe the bug or feature before writing code.
2. **Fork and branch** off `main`. Use a descriptive branch name (`fix/avx2-sign-extend`, `feat/wasm-streaming`).
3. **Run checks** before pushing:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --workspace
   ```
4. **Open a pull request** against `main`. Fill in the PR template.

## What's in scope

- Performance improvements to SIMD kernels (`needle-core`)
- New platform targets (e.g. RISC-V, WASM SIMD128)
- Additional language bindings via the C ABI
- Improved test coverage (more parity examples, edge cases)
- Documentation improvements
- CI / tooling improvements

## What's out of scope

- Changes to the model architecture or weights (those belong in [cactus-compute/needle](https://github.com/cactus-compute/needle))
- Training code or Python pipeline changes

## Code style

- `cargo fmt` for formatting (enforced in CI)
- `cargo clippy -- -D warnings` must pass
- Comments only where the *why* is non-obvious; no obvious comments

## Discussions

For broader questions, use [GitHub Discussions](https://github.com/geekgineer/needle-rs/discussions).
