# CLI quickstart

Runs the needle-rs CLI against a weather tool definition.

## Prerequisites

- Rust toolchain (`cargo build --release`)
- Model weights at `weights/needle.safetensors` and `weights/vocab.txt`
  (export with `python tools/export.py` or download from HuggingFace)

## Run

```bash
chmod +x run.sh
./run.sh "What is the weather in Tokyo?"
```

Expected output (approximately):

```json
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
```

## Streaming mode

```bash
../../target/release/needle-rs --stream \
  ../../weights/needle.safetensors \
  ../../weights/vocab.txt \
  "Book a flight from London to New York" \
  '[{"name":"book_flight","description":"Book a flight","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}}}}]'
```

Tokens stream to stderr; final JSON goes to stdout.
