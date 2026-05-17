<div align="center">
  <img src="assets/logo.svg" alt="needle-rs" width="380"/>
  <br/><br/>
  <p><strong>A working AI agent in 258 KB of WebAssembly.</strong></p>
  <p>
    <a href="https://needle-rs.pages.dev">Live demo</a> ·
    <a href="#quick-start">Quick start</a> ·
    <a href="#how-it-works">How it works</a> ·
    <a href="https://github.com/cactus-compute/needle">Original model</a>
  </p>
  <br/>
  <a href="https://github.com/geekgineer/needle-rs/actions/workflows/ci.yml"><img src="https://github.com/geekgineer/needle-rs/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://crates.io/crates/needle-rs"><img src="https://img.shields.io/crates/v/needle-rs.svg" alt="crates.io"/></a>
  <a href="https://www.npmjs.com/package/needle-rs"><img src="https://img.shields.io/npm/v/needle-rs.svg" alt="npm"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"/></a>
</div>

---

`needle-rs` runs a 26M-parameter tool-calling transformer entirely client-side. No Python, no server, no API key. The whole runtime is a 258 KB WebAssembly module; weights are 22 MB. Drop it into a webpage and your users get function calling that never leaves their device.

The model is [Needle](https://github.com/cactus-compute/needle) by [Cactus Compute](https://github.com/cactus-compute) — a single-shot tool router that maps `(query, tool list) → JSON call`. We built the runtime to deploy it where the official Python implementation can't go: browsers, embedded systems, edge workers, anything with a WASM engine or a C ABI.

## Why this matters

Most "tool-calling" stacks ship a 10 MB JavaScript SDK that calls a 100 GB model over the network. `needle-rs` ships the whole agent.

| Stack | Wire size | Latency | Cost per call | Privacy |
|---|---|---|---|---|
| OpenAI function calling | API call | 300–800 ms | $$ per token | data leaves device |
| Local llama.cpp + 1B model | 700 MB+ | varies | free | local |
| ONNX Runtime Web + your model | 8 MB runtime + model | varies | free | local |
| **`needle-rs` + Needle** | **258 KB + 22 MB** | **~280 ms** | **free** | **local** |

Same answer to "did the user ask for a flight booking?" — at a fraction of the footprint.

## Quick start

**Browser**

```bash
npm install needle-rs
```

```js
import init, { NeedleWasm } from "needle-rs";

await init();
const engine = NeedleWasm.load(weights, vocab);

engine.run(
  "Book a flight from London to JFK tomorrow",
  JSON.stringify([{ name: "book_flight", parameters: { origin: "string", destination: "string", date: "string" } }])
);
// → {"name":"book_flight","arguments":{"origin":"London","destination":"JFK","date":"tomorrow"}}
```

**Rust**

```bash
cargo add needle-infer
```

```rust
use needle_infer::NeedleEngine;

let engine = NeedleEngine::load("needle.safetensors", "vocab.txt")?;
let result = engine.run(query, tools_json);
println!("{}", result.text);
```

**Weights**

```bash
huggingface-cli download Abdalrahman/needle-rs-safetensors \
  needle.safetensors vocab.txt --local-dir weights/
```

Or load directly from a URL in the browser — no install step.

## Where it runs

| Target | Status | Binary |
|---|:---:|---|
| Browser (WASM) | ✓ | 258 KB |
| Node.js (WASM) | ✓ | 258 KB |
| Cloudflare Workers | ✓ | 258 KB |
| Linux / macOS / Windows CLI | ✓ | 533 KB |
| C / C++ / Python / Go / Swift (FFI) | ✓ | 557 KB shared lib |
| `no_std` embedded (Rust) | ✓ | size varies |
| iOS / Android | use [Cactus](https://github.com/cactus-compute/cactus) | — |
| Apple NPU / Snapdragon NPU | use [Cactus](https://github.com/cactus-compute/cactus) | — |

Cactus's official engine targets mobile and NPUs with hand-tuned ARM SIMD. `needle-rs` targets everywhere else. The two stacks are complementary.

## How it works

Needle is a 26M-parameter encoder-decoder transformer with a small twist: it's trained to do exactly one thing — emit a function-call JSON object from a query and a tool list. That focus is why a model this small works at all.

- **Encoder–decoder SAN.** The encoder reads the query and tool definitions once. The decoder generates the output JSON token by token, attending to the encoder's cached KV. Single forward pass per call.
- **INT4 quantization.** All attention and FFN weights are stored as packed 4-bit nibbles with per-32-row scales. Matvec dequantizes on the fly — the full f32 weight matrix is never materialized. AVX2 on x86_64, NEON on aarch64, scalar fallback for WASM.
- **Constrained decoding.** A character-level trie over valid tool names and argument keys, plus a three-state JSON machine, masks logits at every step. The output is always syntactically valid JSON pointing at a real tool — never a hallucinated function name, never broken syntax.
- **Two schema formats.** Accepts both the flat `{ "location": { "type": "string" } }` style and OpenAI's `{ "type": "object", "properties": { ... } }` style. The Python reference handles only the flat form.

Architecture deep-dive: [ARCHITECTURE.md](ARCHITECTURE.md). Parity with the Python reference is verified by 560 token-level test vectors plus 55 constrained-decoder unit tests.

## API

<details>
<summary><strong>JavaScript / TypeScript</strong></summary>

```js
engine.run(query, tools)                              // → string
engine.run_stream(query, tools, (id, piece) => {})    // per-token callback → final string
engine.run_batch([{ query, tools }, ...])             // → string[]
engine.encode_contrastive(text)                       // → Float32Array | null
engine.retrieve_tools(query, descriptionsJson, topK)  // semantic tool routing
```
</details>

<details>
<summary><strong>Rust</strong></summary>

```rust
engine.run(query, tools_json);
engine.run_stream(query, tools_json, |_id, piece| print!("{piece}"));
engine.run_batch(&[(q1, t1), (q2, t2)]);
engine.encode_contrastive(text);            // → Option<Vec<f32>>
engine.retrieve_tools(query, descs, k);     // → Vec<(usize, f32)>
```
</details>

<details>
<summary><strong>C (and anything with FFI)</strong></summary>

```c
#include "needle.h"

NeedleHandle h  = needle_load("needle.safetensors", "vocab.txt");
const char *out = needle_run(h, query, tools_json);
printf("%s\n", out);
needle_free_str((char *)out);
needle_free(h);
```

Full header: [`crates/needle-c/include/needle.h`](crates/needle-c/include/needle.h). Null-safe throughout; errors via thread-local `needle_last_error()`.
</details>

## Benchmarks

Intel i7-1185G7 (Tiger Lake, 4-core), Linux, release build, median of 5 runs.

| | |
|---|---|
| End-to-end (load + infer) | **283 ms** |
| Warm inference only | **~80 ms** |
| INT4 matvec 512×512 (AVX2) | **83 µs · 3.2 Gelem/s** |
| INT4 matvec 2048×512 (AVX2) | **311 µs · 3.1 Gelem/s** |

Apple Silicon NEON path is implemented but unbenchmarked — M-series numbers welcome via PR.

Stripped release sizes:

| | |
|---|---|
| WASM module | **258 KB** |
| CLI binary | **533 KB** |
| C shared library | **557 KB** |
| Weights (INT4 SafeTensors) | **22 MB** |
| Runtime dependencies | **1** (libm; WASM build adds wasm-bindgen) |

Full methodology and raw numbers: [BENCHMARKS.md](BENCHMARKS.md).

## What it's good for

- **Browser-side intent routing.** Decide which API to call before making the network request. Sub-second, zero servers.
- **Edge function dispatch.** Tool calling inside Cloudflare Workers, Vercel Edge, Deno Deploy — anywhere with a WASM runtime.
- **On-device privacy.** User queries never leave the browser tab. Useful for healthcare, legal, and any context where sending text to OpenAI is a non-starter.
- **Embedded agents.** `no_std` core means the kernels run on microcontrollers with enough RAM for the weights.

What it's *not* good for: open-ended chat, long-context reasoning, anything where you'd reach for a >1B-parameter model. Needle is a router, not a generalist.

## Acknowledgements

Needle is designed and trained by [Henry Ndubuaku](https://github.com/hndubuaku) and the [Cactus Compute](https://github.com/cactus-compute) team. The model architecture, training code, dataset, and weights are entirely their work, released under MIT. `needle-rs` is an independent Rust runtime — no upstream code is copied, only the published architecture is implemented.

If you find this useful, please star the [upstream Needle repo](https://github.com/cactus-compute/needle) as well.

## Citation

```bibtex
@software{needle2026,
  author  = {Ndubuaku, Henry and {Cactus Compute}},
  title   = {Needle: A 26M-Parameter Tool-Calling Transformer},
  year    = {2026},
  url     = {https://github.com/cactus-compute/needle},
  license = {MIT}
}

@software{needlers2026,
  author  = {Ibrahim, Abdalrahman},
  title   = {needle-rs: Pure-Rust WASM Runtime for Needle},
  year    = {2026},
  url     = {https://github.com/geekgineer/needle-rs},
  license = {MIT}
}
```

---

MIT — see [LICENSE](LICENSE).