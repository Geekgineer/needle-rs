# Python via C FFI

Run Needle inference from Python using `ctypes` — no JAX, no PyTorch, no ML dependencies.
Only the Python standard library and the compiled `libneedle_c` shared library are required.

## Build the C library

```bash
cargo build --release -p needle-c
```

This produces `target/release/libneedle_c.so` (Linux), `libneedle_c.dylib` (macOS),
or `needle_c.dll` (Windows).

## Run

```bash
python infer.py \
  --query "What is the weather in Berlin?" \
  --tools '[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
```

Expected output:

```json
{"name": "get_weather", "arguments": {"location": "Berlin"}}
```

## Streaming

```bash
python infer.py --stream --query "Book a flight from London to New York" \
  --tools '[{"name":"book_flight","description":"Book a flight","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}}}}]'
```

## What this demonstrates

- Zero-Python-ML-dependency inference: the heavy lifting is in the Rust shared library
- `ctypes` bindings to the C ABI (`needle_load`, `needle_run`, `needle_run_stream`, `needle_free`)
- Streaming callback: a Python function fires per token
- Works with Python 3.8+ on Linux, macOS, and Windows

For the full C API reference, see `crates/needle-c/include/needle.h` and `docs/c-ffi.md`.
