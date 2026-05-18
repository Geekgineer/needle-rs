# HuggingFace Model Card

**Filename:** `README.md` in the `Abdalrahman/needle-rs-safetensors` HuggingFace repository.

## Upload checklist

1. **Upload `banner.svg` first** (from `assets/banner.svg` in this repo) — commit it to the HF
   repo before the README. If you commit the README first, the image will 404 on first render
   and HF's CDN cache may stick on the broken state for hours.
2. Verify `Cactus-Compute/needle` matches Cactus's actual HF org slug exactly
   (`https://huggingface.co/Cactus-Compute/needle` — checked, returns 200 as of 2026-05-17).
   If they ever rename or move the repo, update `base_model:` below before publishing.
3. `inference: false` is correct — there is no HF Inference-compatible adapter.
4. `library_name: needle-rs` is not a registered HF library; HF will display it as-is.
   That is fine — it links users to the runtime.

## Paste below into HF README.md (everything from the `---` onward, verbatim)

The YAML front-matter must be at the very top of the file with no blank line before it.

---

---
license: mit
language:
  - en
library_name: needle-rs
tags:
  - tool-calling
  - function-calling
  - rust
  - wasm
  - webassembly
  - on-device
  - edge-ai
  - quantized
  - int4
  - safetensors
  - no-server
pipeline_tag: text-generation
base_model: Cactus-Compute/needle
base_model_relation: quantized
inference: false
---

<div align="center">
  <img src="./banner.svg" alt="needle-rs" width="100%"/>
</div>

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Geekgineer%2Fneedle--rs-181717?style=flat-square&logo=github)](https://github.com/Geekgineer/needle-rs)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-needle--rs.pages.dev-CE422B?style=flat-square)](https://needle-rs.pages.dev)
[![npm](https://img.shields.io/npm/v/needle-rs?style=flat-square&color=CE422B)](https://www.npmjs.com/package/needle-rs)
[![PyPI](https://img.shields.io/pypi/v/needle-rs?style=flat-square&color=CE422B)](https://pypi.org/project/needle-rs/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://github.com/Geekgineer/needle-rs/blob/main/LICENSE)

</div>

# needle-rs-safetensors

**INT4-packed SafeTensors weights for [Needle](https://github.com/cactus-compute/needle), ready to load into the [needle-rs](https://github.com/Geekgineer/needle-rs) pure-Rust + WebAssembly runtime.**

> **This is a format conversion.** The model itself — its architecture, training procedure, dataset, and original weights — is the work of [**Cactus Compute**](https://github.com/cactus-compute) (Henry Ndubuaku et al., 2026), released under MIT. Original repository: [`Cactus-Compute/needle`](https://huggingface.co/Cactus-Compute/needle).
>
> If you build with these weights, you are building on Cactus's Needle. Please credit them in any publication, blog post, product, or downstream model that incorporates this work. Citation template below.

## Quick links

- **Original model:** [`Cactus-Compute/needle`](https://huggingface.co/Cactus-Compute/needle) — upstream weights, training code, paper
- **Runtime:** [`geekgineer/needle-rs`](https://github.com/geekgineer/needle-rs) — Rust, WASM, C ABI
- **Live demo:** [needle-rs.pages.dev](https://needle-rs.pages.dev) — runs entirely in your browser
- **Weight format spec:** [ARCHITECTURE.md](https://github.com/geekgineer/needle-rs/blob/main/ARCHITECTURE.md)

## Files

| File | Size | Description |
|---|---|---|
| `needle.safetensors` | 22 MB | INT4-packed attention/FFN weights + BF16 norms |
| `vocab.txt` | 120 KB | 8,192 SentencePiece pieces (TSV: `piece\tscore`) |
| `banner.svg` | 6 KB | Repo banner |

## Model summary

| | |
|---|---|
| Architecture | Encoder–decoder transformer (SAN) |
| Parameters | 26M |
| Hidden size | 512 |
| Encoder / decoder layers | 12 / 8 |
| Attention heads (Q / KV) | 8 / 4 (GQA, repeat=2) |
| Vocabulary | 8,192 (SentencePiece BPE) |
| Max encoder length | 1,024 tokens |
| Quantization | INT4 group-wise (`group_size=32`) for attention + FFN; BF16 for norms/embeddings |
| Output | Structured JSON tool calls |

For full architectural details, training procedure, and benchmarks, see the [upstream Needle repository](https://github.com/cactus-compute/needle).

## Weight format

The SafeTensors file uses a custom `I4` dtype for quantized kernels:

- **Group-wise INT4** with `group_size=32`, per-group scale = `max|w| / 7`, packed as nibbles (low nibble = even row, high nibble = odd row, per output column).
- **Non-kernel parameters** (RMSNorm γ, gate vectors, embeddings) stored in BF16.
- **Model config** (d_model, num_heads, max_seq_len, etc.) embedded in the SafeTensors `__metadata__` JSON, so no separate config file is needed.

This format is consumed directly by `needle-rs`. It is not compatible with `transformers`, `safetensors-rust` direct loading without the `needle-rs` engine, or other generic SafeTensors consumers, because the `I4` dtype is non-standard.

## How to use

The intended runtime is [`needle-rs`](https://github.com/geekgineer/needle-rs). The same weights work across all its deployment targets — native CLI, Rust API, C FFI, and browser/Node.js via WebAssembly.

### Download weights

```python
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download("Abdalrahman/needle-rs-safetensors", "needle.safetensors")
vocab_path   = hf_hub_download("Abdalrahman/needle-rs-safetensors", "vocab.txt")
```

Or via CLI:

```bash
huggingface-cli download Abdalrahman/needle-rs-safetensors \
  needle.safetensors vocab.txt --local-dir weights/
```

### Command line

```bash

# Single inference
./needle-rs weights/needle.safetensors weights/vocab.txt \
  "What's the weather in Paris?" \
  '[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
# → {"name":"get_weather","arguments":{"location":"Paris"}}
```

### Rust

```rust
use needle_infer::NeedleEngine;

let engine = NeedleEngine::load(
    "weights/needle.safetensors",
    "weights/vocab.txt",
)?;
let result = engine.run(query, tools_json);
println!("{}", result.text);
```

### Browser (WebAssembly)

```js
import init, { NeedleWasm } from "needle-rs";

await init();

const HF = "https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main";

const [weights, vocab] = await Promise.all([
  fetch(`${HF}/needle.safetensors`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
  fetch(`${HF}/vocab.txt`).then(r => r.text()),
]);

const engine = NeedleWasm.load(weights, vocab);
const result = engine.run("Book a flight from London to JFK tomorrow", toolsJson);
// → {"name":"book_flight","arguments":{"origin":"London","destination":"JFK","date":"tomorrow"}}
```

**Live demo:** [needle-rs.pages.dev](https://needle-rs.pages.dev) — the demo loads exactly these files from this repository.

### Python

```bash
pip install needle-rs
```

```python
from needle_rs import NeedleEngine

engine = NeedleEngine.load("weights/needle.safetensors", "weights/vocab.txt")

# Single call
result = engine.run(
    "Book a flight from London to JFK tomorrow",
    '[{"name":"book_flight","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}}}}]',
)
# → [{"name":"book_flight","arguments":{"origin":"London","destination":"JFK","date":"tomorrow"}}]

# Streaming (callback fires per token)
result = engine.run_stream(query, tools_json, lambda token_id, piece: print(piece, end="", flush=True))

# Batch
results = engine.run_batch([("query1", tools1), ("query2", tools2)])

# Semantic tool retrieval (requires weights with a contrastive head)
ranked = engine.retrieve_tools(
    "What's the weather in Paris?",
    ["Get current weather for a location", "Book a flight", "Send an email"],
    top_k=2,
)
# → [(0, 0.91), (1, 0.38)]
```

### Multi-tool routing example

Needle is trained to pick the right tool from a list, not just fill a single tool's parameters:

```bash
./needle-rs weights/needle.safetensors weights/vocab.txt \
  "Turn off the bedroom lights" \
  '[
    {"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}},
    {"name":"play_music","parameters":{"type":"object","properties":{"song":{"type":"string"}}}},
    {"name":"control_lights","parameters":{"type":"object","properties":{"room":{"type":"string"},"state":{"type":"string"}}}},
    {"name":"send_message","parameters":{"type":"object","properties":{"recipient":{"type":"string"},"body":{"type":"string"}}}}
  ]'
# → {"name":"control_lights","arguments":{"room":"bedroom","state":"off"}}
```

## Intended use

- **Client-side intent routing in web applications** — decide which API endpoint to call before issuing the network request, with no server-side LLM.
- **Edge function dispatch** — Cloudflare Workers, Vercel Edge, Deno Deploy, anywhere with a WASM engine and ≤30 MB of available memory.
- **On-device function calling** in privacy-sensitive contexts (healthcare, legal, personal data) where sending user queries to a hosted LLM is unacceptable.
- **Embedded agents** on hardware with enough RAM for the weights (≈30 MB working set including activations).
- **Tool retrieval** — the optional contrastive head exposed by `needle-rs.encode_contrastive()` can semantically rank a tool catalogue before passing the top-K to the generator.

## Limitations

- **Tool calling only.** Needle is trained for the single task of mapping a query plus tool definitions to a JSON call. It is not a chat model and will not produce meaningful free-form text.
- **Single-shot.** No multi-turn dialogue, no chain-of-thought, no tool-use feedback loop. Each call is independent.
- **English-trained.** Multilingual behavior is not evaluated by upstream and is not guaranteed.
- **Greedy decoding only** in `needle-rs` — temperature and sampling are intentionally not supported, since stochasticity is undesirable for routing.
- **Encoder length ≤ 1,024 tokens.** Long tool catalogues must be pre-filtered via contrastive retrieval before being passed in.
- **Small-model failure modes apply.** Ambiguous queries, tools with overlapping descriptions, or unusual parameter schemas can produce unexpected routings. The constrained decoder guarantees syntactic validity, not semantic correctness.

## Out of scope

- General-purpose text generation, summarization, translation, or chat.
- Long-context reasoning (>1,024 tokens of input).
- Reasoning over tool outputs (the model produces calls, not results — your application executes the call and decides what to do with the response).
- Production use in safety-critical domains without an evaluation suite covering the specific tool catalogue and query distribution.

## Citation

If you publish or distribute work that uses these weights, please cite **the upstream Needle paper/repository**:

```bibtex
@misc{ndubuaku2026needle,
  title  = {Needle: A 26M-Parameter Tool-Calling Transformer},
  author = {Ndubuaku, Henry and Mroz, Jakub and Mosoyan, Karen and Shemet, Roman
            and Sandhu, Parkirat and Kumar, Satyajit and Cylich, Noah and Lee, Justin H.},
  year   = {2026},
  url    = {https://github.com/cactus-compute/needle}
}
```

Optionally, cite the runtime if relevant to your work:

```bibtex
@misc{ibrahim2026needlers,
  title  = {needle-rs: Pure-Rust + WebAssembly Runtime for Needle},
  author = {Ibrahim, Abdalrahman},
  year   = {2026},
  url    = {https://github.com/geekgineer/needle-rs}
}
```

## License

MIT — matching the upstream Needle release.

This repository performs only **format conversion** (Flax/Pickle → SafeTensors with INT4 packing) and quantization (BF16 → INT4 group-wise) of weights originally released by Cactus Compute under MIT. No retraining, fine-tuning, distillation, or modification of model behavior has been performed. All learned parameters originate from the upstream release.

## Acknowledgments

The Needle model is the work of [Henry Ndubuaku](https://github.com/hndubuaku) and the [Cactus Compute](https://github.com/cactus-compute) team. Their decision to release the weights, training code, and dataset generation pipeline under MIT is what makes downstream runtimes like `needle-rs` possible. If this conversion is useful to you, please consider [starring the upstream repository](https://github.com/cactus-compute/needle) as well.