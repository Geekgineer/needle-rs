# HuggingFace Model Card

**Filename:** `README.md` in the `Abdalrahman/needle-rs-safetensors` HuggingFace repository.

## Upload checklist

1. **Upload `banner.svg` first** (from `assets/banner.svg` in this repo) — commit it to the HF
   repo before the README. If you commit the README first, the image will 404 on first render
   and HF's CDN cache may stick on the broken state for hours.
2. Verify `Cactus-Compute/needle` matches Cactus's actual HF org slug exactly
   (`https://huggingface.co/Cactus-Compute/needle` — checked, returns 200 as of 2026-08-20).
   If they ever rename or move the repo, update `base_model:` below before publishing.
   **These weights convert Needle v1.** Upstream has since released v2 at
   `Cactus-Compute/needle2` and v3 at `Cactus-Compute/needle3`, each a different
   architecture in a different container (see `docs/v2-port-record.md` and
   `docs/v3-port-record.md`). All three repos resolve; keep `base_model:` on
   `needle` so the card is not mistaken for a v2 or v3 conversion.
   Note that **neither v2 nor v3 needs a conversion repo at all** — needle-rs
   reads upstream's own `.cact` containers directly, so there is nothing to host
   and nothing to keep in sync. This repository exists only because v1 shipped
   as a Flax checkpoint. Say so on the card rather than letting readers assume a
   v2 or v3 equivalent is missing.
3. `inference: false` is correct — there is no HF Inference-compatible adapter.
4. `library_name: needle-rs` is not a registered HF library; HF will display it as-is.
   That is fine — it links users to the runtime.

## The live card can be ahead of this file

**Pull the published card before editing it.** This copy has drifted behind the
live one before: on the 0.3.0 update the HF page had a clearer title, a
`needle-v1` tag and a "Looking for Needle v2?" callout that this file did not,
so pasting this file over it would have deleted all three. The published page is
the authority; this file is a working copy that is only trustworthy just after a
sync.

```bash
curl -sL https://huggingface.co/Abdalrahman/needle-rs-safetensors/raw/main/README.md \
  -o /tmp/live.md
diff /tmp/live.md <(sed -n '/^---$/,$p' docs/hf-model-card.md)   # reconcile before editing
```

Edit the live copy, upload it, then sync this file back to exactly what was
published.

## Paste below into HF README.md

Copy from the `---` that begins `license: mit` — **not** the horizontal rule
below this paragraph — through to the end of the file, verbatim. The YAML
front-matter must be the very first line of the HF README with no blank line
above it; a stray rule or blank line there makes HF render the block as text
instead of parsing it.

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
  - needle-v1
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

# needle-rs-safetensors — Needle **v1** weights

**INT4-packed SafeTensors weights for [Needle v1](https://github.com/cactus-compute/needle) — 26M parameters, encoder–decoder — ready to load into the [needle-rs](https://github.com/Geekgineer/needle-rs) pure-Rust + WebAssembly runtime.**

> ### Looking for Needle 3 or Needle 2?
> **`needle-rs` runs all three generations.** Neither newer generation needs a
> conversion, and neither is hosted here — point the runtime straight at
> upstream's containers. This repository exists only because **v1** shipped as
> Flax/Pickle, which no Rust runtime can load.
>
> | | Weights | Loader |
> |---|---|---|
> | **v3** (121M, decoder-only, reasons first) | [`Cactus-Compute/needle3`](https://huggingface.co/Cactus-Compute/needle3) — `needle3.cact` | `V3Engine` / `NeedleV3Wasm` |
> | **v2** (45M, decoder-only) | [`Cactus-Compute/needle2`](https://huggingface.co/Cactus-Compute/needle2) — `needle2.cact` | `V2Engine` / `NeedleV2Wasm` |
> | **v1** (26M, encoder–decoder) | **this repo** — `needle.safetensors` + `vocab.txt` | `NeedleEngine` / `NeedleWasm` |
>
> All three run in the [live demo](https://needle-rs.pages.dev) — switch generation in the browser.

> **This is a format conversion.** The model itself — its architecture, training procedure, dataset, and original weights — is the work of [**Cactus Compute**](https://github.com/cactus-compute) (Henry Ndubuaku et al., 2026). The Needle 1 weights converted here are MIT, as is Needle 2; the upstream repository and the Needle 3 weights are Apache-2.0. Original repository: [`Cactus-Compute/needle`](https://huggingface.co/Cactus-Compute/needle).
>
> If you build with these weights, you are building on Cactus's Needle. Please credit them in any publication, blog post, product, or downstream model that incorporates this work. Citation template below.

## Quick links

- **Original model:** [`Cactus-Compute/needle`](https://huggingface.co/Cactus-Compute/needle) — upstream weights, training code, paper
- **Runtime:** [`geekgineer/needle-rs`](https://github.com/geekgineer/needle-rs) — Rust, WASM, C ABI
- **Live demo:** [needle-rs.pages.dev](https://needle-rs.pages.dev) — runs entirely in your browser
- **Weight format spec:** [ARCHITECTURE.md](https://github.com/geekgineer/needle-rs/blob/main/ARCHITECTURE.md)

## Running Needle 3 or Needle 2 instead

`needle-rs` >= 0.3.0 reads upstream's `.cact` containers directly — one runtime,
all three generations:

```bash
hf download Cactus-Compute/needle3 needle3.cact --local-dir weights/

needle-rs --json weights/needle3.cact "What's the weather in Paris?" \
  '[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
# → [{"name":"get_weather","arguments":{"location":"Paris"}}]
```

One runtime, three generations — `NeedleEngine` / `NeedleWasm` for v1 here,
`V2Engine` / `NeedleV2Wasm` for v2, and `V3Engine` / `NeedleV3Wasm` for v3.
**Needle 3 is upstream's current release** and where development continues:
121M parameters, an 8192-token context, and a `<think>` reasoning step before
the call. Prefer it for new work. Needle 2 remains the smallest viable browser
deployment, and these v1 weights are for when you specifically want the
smallest model.

## Files

| File | Size | Description |
|---|---|---|
| `needle.safetensors` | 22.3 MB | INT4-packed attention/FFN weights + BF16 norms |
| `vocab.txt` | 122 kB | 8,192 SentencePiece pieces (TSV: `piece\tscore`) |
| `config.json` | 320 B | Geometry, for tooling that expects a config file |
| `banner.svg` | 5.3 kB | Repo banner |

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
- **Model config** ships as a separate `config.json` (the SafeTensors header carries no `__metadata__` block). `needle-rs` does not read it — the engine derives geometry from tensor shapes — but it is there for tooling that expects one.

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
hf download Abdalrahman/needle-rs-safetensors \
  needle.safetensors vocab.txt --local-dir weights/
```

### Command line

```bash

# Single inference
./needle-rs weights/needle.safetensors weights/vocab.txt \
  "What's the weather in Paris?" \
  '[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]'
# → [{"name":"get_weather","arguments":{"location":"Paris"}}]
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
// → [{"name":"book_flight","arguments":{"origin":"London","destination":"JFK","date":"tomorrow"}}]
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

# Semantic tool retrieval. NOTE: the weights in THIS repository carry no
# contrastive head (339 tensors, none of them a projection head), so this
# returns [] and encode_contrastive() returns None. The API is here for
# checkpoints that do have the head.
ranked = engine.retrieve_tools(
    "What's the weather in Paris?",
    ["Get current weather for a location", "Book a flight", "Send an email"],
    top_k=2,
)
# → []  with these weights
```

### Multi-tool routing example

Needle is trained to pick the right tool from a list, not just fill a single tool's parameters:

```bash
TOOLS='[
  {"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}},
  {"name":"play_music","parameters":{"type":"object","properties":{"song":{"type":"string"}}}},
  {"name":"send_message","parameters":{"type":"object","properties":{"recipient":{"type":"string"},"body":{"type":"string"}}}}
]'

./needle-rs weights/needle.safetensors weights/vocab.txt "What's the weather in Paris?" "$TOOLS"
# → [{"name":"get_weather","arguments":{"location":"Paris"}}]

./needle-rs weights/needle.safetensors weights/vocab.txt "Play Yesterday" "$TOOLS"
# → [{"name":"play_music","arguments":{"song":"Yesterday"}}]
```

An empty array — `[]` — is a deliberate abstention, not a failure to parse: the
model declined to route. v1 abstains readily as the catalogue grows or the
phrasing drifts, which is a real limitation of the 26M model rather than a bug in
the runtime. See [Limitations](#limitations).

## Intended use

- **Client-side intent routing in web applications** — decide which API endpoint to call before issuing the network request, with no server-side LLM.
- **Edge function dispatch** — Cloudflare Workers, Vercel Edge, Deno Deploy, anywhere with a WASM engine and ≤30 MB of available memory.
- **On-device function calling** in privacy-sensitive contexts (healthcare, legal, personal data) where sending user queries to a hosted LLM is unacceptable.
- **Embedded agents** on hardware with enough RAM for the weights (≈30 MB working set including activations).
- **Tool retrieval** — `needle-rs` exposes `encode_contrastive()` / `retrieve_tools()` for ranking a large tool catalogue before passing the top-K to the generator. This needs a checkpoint with a contrastive head; **the weights in this repository do not have one**, so both return empty on these files.

## Limitations

- **Tool calling only.** Needle is trained for the single task of mapping a query plus tool definitions to a JSON call. It is not a chat model and will not produce meaningful free-form text.
- **Single-shot.** No multi-turn dialogue, no chain-of-thought, no tool-use feedback loop. Each call is independent.
- **English-trained.** Multilingual behavior is not evaluated by upstream and is not guaranteed.
- **Greedy decoding only** on the v1 path in `needle-rs` — stochasticity is undesirable for routing, so no sampling is exposed. (`--temperature` and `--seed` exist, but apply to the v3 and v2 paths.)
- **Encoder length ≤ 1,024 tokens.** Long tool catalogues must be truncated or pre-filtered before being passed in.
- **Routing degrades as the catalogue grows.** Measured on these weights: with
  three tools, clean queries route correctly; with four, several queries that a
  human would find unambiguous return `[]` instead, and one produced a malformed
  call with a repeated argument key. Keep the catalogue small, or use Needle 3 or 2,
  which handled the same four-tool cases correctly.
- **No contrastive head in this checkpoint**, so the retrieval API cannot be used
  to do that pre-filtering with these weights.
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

MIT — matching the Needle 1 weights this repository converts.

This repository performs only **format conversion** (Flax/Pickle → SafeTensors with INT4 packing) and quantization (BF16 → INT4 group-wise) of weights originally released by Cactus Compute under MIT. No retraining, fine-tuning, distillation, or modification of model behavior has been performed. All learned parameters originate from the upstream release.

Note that licensing differs by generation upstream: Needle 1 and Needle 2 weights are MIT, while the upstream repository and the Needle 3 weights are Apache-2.0. Check the licence on the generation you ship.

## Acknowledgments

The Needle model is the work of [Henry Ndubuaku](https://github.com/hndubuaku) and the [Cactus Compute](https://github.com/cactus-compute) team. Their decision to release the weights, training code and dataset generation pipeline openly — Needle 1 and Needle 2 under MIT, the repository and Needle 3 under Apache-2.0 — is what makes downstream runtimes like `needle-rs` possible. If this conversion is useful to you, please consider [starring the upstream repository](https://github.com/cactus-compute/needle) as well.
