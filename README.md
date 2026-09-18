<div align="center">
  <img src="https://raw.githubusercontent.com/Geekgineer/needle-rs/main/assets/banner.svg" alt="needle-rs" width="100%"/>

  <br/><br/>

  <p>
    <a href="https://needle-rs.pages.dev"><b>→ Live demo</b></a>
  </p>

  <p>
    <a href="https://github.com/Geekgineer/needle-rs/actions/workflows/ci.yml"><img src="https://github.com/Geekgineer/needle-rs/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"/></a>
    <a href="https://github.com/Geekgineer/needle-rs/actions/workflows/release.yml"><img src="https://github.com/Geekgineer/needle-rs/actions/workflows/release.yml/badge.svg" alt="Release"/></a>
    <a href="https://github.com/Geekgineer/needle-rs/actions/workflows/wasm-demo.yml"><img src="https://github.com/Geekgineer/needle-rs/actions/workflows/wasm-demo.yml/badge.svg?branch=main" alt="Demo"/></a>
    <a href="#parity"><img src="https://img.shields.io/badge/parity-token--exact-brightgreen?style=flat-square" alt="Token-exact parity"/></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/Needle-v1%20%2B%20v2%20%2B%20v3-CE422B?style=flat-square" alt="Needle v1, v2 and v3"/></a>
    <a href="https://crates.io/crates/needle-infer"><img src="https://img.shields.io/crates/v/needle-infer?style=flat-square&color=CE422B" alt="crates.io"/></a>
    <a href="https://www.npmjs.com/package/needle-rs"><img src="https://img.shields.io/npm/v/needle-rs?style=flat-square&color=CE422B" alt="npm"/></a>
    <a href="https://pypi.org/project/needle-rs/"><img src="https://img.shields.io/pypi/v/needle-rs?style=flat-square&color=CE422B" alt="PyPI"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT"/></a>
  </p>

  <p>
    <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
    <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
    <a href="#parity">Parity</a> &nbsp;·&nbsp;
    <a href="https://github.com/cactus-compute/needle">Upstream model</a>
  </p>
</div>

<br/>

<img src="https://raw.githubusercontent.com/Geekgineer/needle-rs/main/assets/terminal.svg" alt="A needle-rs session: Needle v3 reasoning before a tool call, the same query with the int8 cache, then Needle v2 from the same binary" width="100%"/>

<p align="center">
  <sub>Real output. For the browser version, try the <a href="https://needle-rs.pages.dev">live demo</a> — it runs all three generations.</sub>
</p>

<br/>

A pure-Rust + WebAssembly runtime for [Needle](https://github.com/cactus-compute/needle) by [Cactus Compute](https://github.com/cactus-compute) — small transformers that map `(query, tool list)` to a JSON function call. Deploys to browsers, edge workers, CLIs, Python, and `no_std` embedded targets. No server, no API key, no data leaving the device.

**All three model generations are supported in parallel:** Needle **3** (121M, one 35.3 MB `.cact` file), Needle **2** (45M, 13.7 MB `.cact`) and Needle **1** (26M, SafeTensors + vocabulary). Same runtime, same API shape, one binary — the generation comes from the container, not the file name.

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="why-this-matters">
  <img src="https://img.shields.io/badge/-Why_this_matters-CE422B?style=flat-square" height="22" alt="Why this matters"/>
</h2>

Tool calling usually means a paid API round-trip or hundreds of megabytes on disk. This ships the whole agent in **14 MB** with Needle 2, or 36 MB with Needle 3, and runs it in a browser tab.

<table>
<thead>
<tr><th align="left">Stack</th><th align="right">Deploy size</th><th align="right">Cost</th><th align="center">Privacy</th><th align="center">Offline</th></tr>
</thead>
<tbody>
<tr><td>Hosted function calling</td><td align="right">SDK + API</td><td align="right">$ per token</td><td align="center">leaves device</td><td align="center">✗</td></tr>
<tr><td>llama.cpp + a 1B local model</td><td align="right">700 MB+</td><td align="right">free</td><td align="center">local</td><td align="center">✓</td></tr>
<tr><td>ONNX Runtime Web + a model</td><td align="right">8 MB + model</td><td align="right">free</td><td align="center">local</td><td align="center">✓</td></tr>
<tr><td><b><code>needle-rs</code> + Needle 3</b></td><td align="right"><b>537 KB + 35.3 MB</b></td><td align="right"><b>free</b></td><td align="center"><b>local</b></td><td align="center"><b>✓</b></td></tr>
<tr><td><code>needle-rs</code> + Needle 2 <sub>(smallest)</sub></td><td align="right">537 KB + 13.7 MB</td><td align="right">free</td><td align="center">local</td><td align="center">✓</td></tr>
</tbody>
</table>

The runtime is 537 KB of WebAssembly (195 KB over the wire, brotli) with **one** runtime dependency, and carries all three model generations. A Needle 2 session needs about 23 MB of working memory; a Needle 3 session adds 8.8 MB of key/value cache at 512 tokens, or 2.3 MB with `--kv-int8` — see [choosing a generation](#generations).

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="generations">Choosing a generation</h2>

All three run from the same binary, the same WASM module and the same Python
package. The runtime picks the right engine from the container itself, so
nothing depends on a file name.

| | Needle 3 | Needle 2 | Needle 1 |
|---|---|---|---|
| Parameters | 121M | 45M | 26M |
| Container | `needle3.cact`, **35.3 MB** | `needle2.cact`, 13.7 MB | `.safetensors` + vocab, 22.3 MB |
| Context | 8192 | 2048 | 1024 |
| KV cache, 512-token session | 8.8 MB, or 2.3 MB at int8 | 3.5 MB | — |
| Reasoning | `<think>` on essentially every query | `<think>` sometimes | none |
| Confidence head | ✓ | ✓ | ✗ |
| Tool retrieval | ✗ | ✓ | ✗ (not in the published weights) |
| Licence | Apache-2.0 | MIT | MIT |

**Pick Needle 3** when quality matters most and you can afford a 35 MB download
and roughly 9 MB of cache — or 2.3 MB with the int8 cache, which is the width
the container declares and costs nothing in answer quality on our tests. It
reasons before answering, which shows up on ambiguous queries and larger tool
catalogues.

**Pick Needle 2** for the smallest viable browser deployment, or when you need
tool retrieval — v3 exports no contrastive head, so `retrieve_tools` and
`encode_contrastive` are absent on the v3 API rather than present and always
empty.

**Needle 1** remains supported for compatibility. It abstains readily as a tool
catalogue grows; prefer a newer generation for new work.

> Needle 3's weights are **Apache-2.0**, where v1 and v2 are MIT. needle-rs
> itself is MIT in every case; the difference applies to the model.

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="quick-start">
  <img src="https://img.shields.io/badge/-Quick_start-CE422B?style=flat-square" height="22" alt="Quick start"/>
</h2>

**Get a model.** v3 and v2 are each a single self-describing file; v1 needs a vocabulary alongside its weights.

```bash
# Needle v3 — weights, geometry and tokenizer in one container
hf download Cactus-Compute/needle3 needle3.cact --local-dir weights/

# Needle v2 — smaller and faster, same one-file story
hf download Cactus-Compute/needle2 needle2.cact --local-dir weights/

# Needle v1
hf download Abdalrahman/needle-rs-safetensors needle.safetensors vocab.txt --local-dir weights/
```

<details open>
<summary><b>CLI</b> &nbsp;—&nbsp; <code>cargo install needle-rs-cli</code></summary>
<br/>

The crate is `needle-rs-cli`; the binary it installs is `needle-rs`. (Both
`needle-cli` and `needle-rs` on crates.io are unrelated projects — don't install
those.)

```bash
# v3 — the container states its own generation, so there is no flag to get wrong
needle-rs --json --constrain weights/needle3.cact \
  "What's the weather in Paris?" \
  '[{"name":"get_weather","parameters":{"type":"object",
     "properties":{"city":{"type":"string"}},"required":["city"]}}]'
# → [{"name":"get_weather","arguments":{"city":"Paris"}}]

# Drop --json to see the <think> block v3 reasons with first.
# --kv-int8 stores the cache at 8 bits: 2.3 MB instead of 8.8 MB at 512 tokens.
needle-rs --kv-int8 weights/needle3.cact "$QUERY" "$TOOLS"

# v2 — same binary, same invocation
needle-rs --json weights/needle2.cact "$QUERY" "$TOOLS"

# v1 — same binary, two files
needle-rs weights/needle.safetensors weights/vocab.txt "$QUERY" "$TOOLS"
```
</details>

<details open>
<summary><b>Rust</b> &nbsp;—&nbsp; <code>cargo add needle-infer</code></summary>
<br/>

```rust
use needle_infer::v3_engine::V3Engine;          // Needle v3
let engine = V3Engine::load("weights/needle3.cact")?;
let text = engine.run(query, tools_json);       // reasoning + call
println!("{}", engine.run_json(query, tools_json).unwrap_or_default());
println!("{:?}", V3Engine::reasoning(&text));   // the <think> block, if any
// Pass the completion, never the bare query — the head scores a finished answer.
println!("{:?}", engine.confidence_for(query, tools_json, &text));

use needle_infer::v2_engine::V2Engine;          // Needle v2
let engine = V2Engine::load("weights/needle2.cact")?;
let out = engine.run(query, tools_json);
println!("{}", out.tool_call.unwrap_or(out.text));

use needle_infer::NeedleEngine;                 // Needle v1
let engine = NeedleEngine::load("weights/needle.safetensors", "weights/vocab.txt")?;
println!("{}", engine.run(query, tools_json).text);
```
</details>

<details open>
<summary><b>Browser / Node.js</b> &nbsp;—&nbsp; <code>npm install needle-rs</code></summary>
<br/>

```js
import init, { NeedleV3Wasm, NeedleV2Wasm, NeedleWasm } from "needle-rs";
await init();

const v3 = NeedleV3Wasm.load(new Uint8Array(cactBytes));
const out = v3.run(query, toolsJson);
v3.run_json(query, toolsJson);                 // just the payload
v3.reasoning(out);                             // the <think> block, or undefined
v3.confidence_for(query, toolsJson, out);      // pass the completion, not the query
v3.kv_bytes(512);                              // 8.8 MB — budget a tab before loading
v3.kv_bytes_int8(512);                         // 2.3 MB at 8 bits
// No retrieve_tools on v3: it exports a confidence head and nothing else.

const v2 = NeedleV2Wasm.load(new Uint8Array(cactBytes));
v2.retrieve_tools(query, descriptions, 3);     // rank tools by relevance

const v1 = NeedleWasm.load(weightsBytes, vocabText);
v1.run(query, toolsJson);
```
</details>

<details open>
<summary><b>Python</b> &nbsp;—&nbsp; <code>pip install needle-rs</code></summary>
<br/>

```python
from needle_rs import V3Engine, V2Engine, NeedleEngine

engine = V3Engine.load("weights/needle3.cact")
engine.run_json(query, tools_json)                     # tool-call payload
engine.generate(query, tools_json, constrain=True)     # dict: text, tool_call,
                                                       # reasoning, stop_reason
engine.generate(query, tools_json, kv_int8=True)       # 8-bit key/value cache
engine.confidence_for(query, tools_json, completion)   # the completion, not the query
engine.kv_bytes(512, kv_int8=True)                     # what a session will cost
# V3Engine has no retrieve_tools — v3 exports no contrastive head.

V2Engine.load("weights/needle2.cact").retrieve_tools(query, descriptions, top_k=3)
NeedleEngine.load("weights/needle.safetensors", "weights/vocab.txt")
```

One `abi3` wheel covers every CPython ≥ 3.8.
</details>

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="versions">
  <img src="https://img.shields.io/badge/-The_three_models-CE422B?style=flat-square" height="22" alt="The three models"/>
</h2>

Upstream replaced v1's encoder–decoder with a decoder-only architecture in a new container, then rebuilt that again for v3 — hybrid local/global attention, a causal convolution over Q/K/V, five Engram sites and a reasoning step. No two generations share weights, loader or quantisation scheme, so `needle-rs` implements all three rather than migrating.

<table>
<thead><tr><th align="left"></th><th align="left">Needle v3</th><th align="left">Needle v2</th><th align="left">Needle v1</th></tr></thead>
<tbody>
<tr><td>Parameters</td><td>121M</td><td>45M</td><td>26M</td></tr>
<tr><td>Architecture</td><td>decoder-only; hybrid local/global attention, QKV conv, 5 Engram sites</td><td>decoder-only; mHC lanes, Engram memory, HadamardMLP</td><td>encoder–decoder SAN</td></tr>
<tr><td>Weights</td><td>Cactus-Quants, ~2.2 bits effective</td><td>Cactus-Quants, ~2.2 bits effective</td><td>symmetric INT4</td></tr>
<tr><td>Files</td><td><b>one</b> <code>.cact</code> — 35.3 MB</td><td><b>one</b> <code>.cact</code> — 13.7 MB</td><td>22 MB + 122 KB vocabulary</td></tr>
<tr><td>Context</td><td>8192</td><td>2048</td><td>1024</td></tr>
<tr><td>Tokenizer</td><td>embedded in the container</td><td>embedded in the container</td><td>separate file</td></tr>
<tr><td>Reasoning trace</td><td>✓ <code>&lt;think&gt;</code> before the call</td><td>—</td><td>—</td></tr>
<tr><td>Constrained decoding</td><td>optional</td><td>optional</td><td>always on</td></tr>
<tr><td>Sampling</td><td>✓ temperature + seed</td><td>✓ temperature + seed</td><td>greedy only</td></tr>
<tr><td>Confidence head</td><td>✓</td><td>✓</td><td>—</td></tr>
<tr><td>Tool retrieval head</td><td>—</td><td>✓ 128-d</td><td>— <sub>architecture has one; the published weights do not</sub></td></tr>
<tr><td>Quantised KV cache</td><td>✓ <code>--kv-int8</code></td><td>—</td><td>—</td></tr>
<tr><td>Weights licence</td><td>Apache-2.0</td><td>MIT</td><td>MIT</td></tr>
</tbody>
</table>

Every example in [`examples/`](examples/) runs on all three.

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="where-it-runs">
  <img src="https://img.shields.io/badge/-Where_it_runs-CE422B?style=flat-square" height="22" alt="Where it runs"/>
</h2>

<table>
<thead><tr><th align="left">Target</th><th align="center">Status</th><th align="right">Binary</th></tr></thead>
<tbody>
<tr><td>Browser / Node.js / Cloudflare Workers <sub>(WASM)</sub></td><td align="center">✓</td><td align="right"><code>537 KB</code> <sub>195 KB over the wire</sub></td></tr>
<tr><td>Linux / macOS / Windows CLI</td><td align="center">✓</td><td align="right"><code>765 KB</code></td></tr>
<tr><td>Python <sub>(abi3 wheel, CPython ≥ 3.8)</sub></td><td align="center">✓</td><td align="right"><code>pip install needle-rs</code></td></tr>
<tr><td>C / C++ / Go / Swift <sub>(FFI)</sub></td><td align="center">✓</td><td align="right"><code>needle_v3_*</code> + <code>needle_v2_*</code> + <code>needle_*</code></td></tr>
<tr><td><code>no_std</code> embedded <sub>(Rust)</sub></td><td align="center">✓</td><td align="right"><sub>size varies</sub></td></tr>
<tr><td>iOS / Android, Apple &amp; Snapdragon NPU</td><td align="center"><sub>use <a href="https://github.com/cactus-compute/cactus">Cactus</a></sub></td><td align="right"><sub>—</sub></td></tr>
</tbody>
</table>

Cactus's own engine targets mobile and NPUs with hand-tuned ARM SIMD. `needle-rs` targets everywhere else. MSRV is **1.87**.

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="how-it-works">
  <img src="https://img.shields.io/badge/-How_it_works-CE422B?style=flat-square" height="22" alt="How it works"/>
</h2>

<table>
<tr>
<td width="32" valign="top" align="center"><sub>1</sub></td>
<td valign="top"><b>Weights are never reconstructed.</b> A Cactus-Quants group dequantises as <code>w = u @ H</code> with <code>H</code> a normalised Walsh–Hadamard matrix. <code>H</code> is symmetric, so <code>dot(x, u @ H) == dot(H @ x, u)</code> — the rotation moves off the weights and onto the activation, paid once per matrix instead of once per row. At 512×512 that is 4 transforms instead of 512, and the inner loop becomes a dot product against packed bytes.</td>
</tr>
<tr>
<td valign="top" align="center"><sub>2</sub></td>
<td valign="top"><b>Fast Walsh–Hadamard, not a matmul.</b> Both places Needle uses <code>H</code> — quantisation groups and HadamardMLP — use a butterfly: <code>n log₂n</code> add/sub instead of <code>n²</code> multiply-accumulates.</td>
</tr>
<tr>
<td valign="top" align="center"><sub>3</sub></td>
<td valign="top"><b>The KV cache is a ring.</b> v2 attends over a 256-token window, so the cache holds 256 positions rather than <code>max_seq_len</code> — 14 MB instead of 113 MB. v3 mixes local and global layers, so its ring is per-layer; <code>--kv-int8</code> stores it at the width the container declares, taking a full-context session from 42.0 MB to 11.2 MB.</td>
</tr>
<tr>
<td valign="top" align="center"><sub>4</sub></td>
<td valign="top"><b>Probe heads stream.</b> Confidence and retrieval pool over every layer's activations at every position — 117 MB if materialised on v2, 504 MB on v3 at full context. An online softmax reaches the same result in 16 KB and 252 KB.</td>
</tr>
<tr>
<td valign="top" align="center"><sub>5</sub></td>
<td valign="top"><b>Constrained decoding.</b> A character trie over declared tool names and argument keys, plus a JSON state machine, masks logits so the payload cannot name a tool that does not exist. Accepts both the flat and OpenAI schema styles.</td>
</tr>
</table>

Architecture deep-dive: [ARCHITECTURE.md](ARCHITECTURE.md) · v2 port record: [docs/v2-port-record.md](docs/v2-port-record.md).

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="parity">
  <img src="https://img.shields.io/badge/-Parity-7EE787?style=flat-square" height="22" alt="Parity"/>
</h2>

The failure mode for a from-scratch reimplementation is silent drift: output that looks right but diverges in the third decimal, producing rare and untraceable bugs. All three engines are held to the reference implementation's exact output.

**Needle v3** — verified against upstream's own model, with the tensor canon pinned by inverting `export._tensors`, since the container's directory is nameless and positional and per-tensor checks alone cannot catch a reordering:

| What | Result |
|---|---|
| Forward pass, 57 positions | max relative deviation **9.0e-6**, **zero** argmax mismatches |
| Incremental decode vs prefill | **bit-identical** (0.000e0) |
| Container: 581 tensors, 196-byte header, codebook | field-for-field match |
| Tokenizer, embedded SentencePiece | exact ids vs `sentencepiece` |
| Engram hash indices | exact as integers |
| Components: MLP **5.4e-6**, attention **2.7e-6**, confidence **2e-6** | — |
| int8 KV cache vs upstream `fake_quant` | exact, and prefill stays bit-identical to decode |

**Needle v2** — verified against upstream's own `decode.forward_cached` running the same weights, reconstructed from the shipped container by [`tools/cact_params.py`](tools/cact_params.py):

| What | Result |
|---|---|
| Forward pass, 788 captured intermediates across 27 layers | max relative deviation **1.9e-5** |
| End to end, 14 prompt/tool combinations (2,482 tokens) | **exact** token ids |
| Container: 145 CQ + 259 FP16 tensors, header, codebook | field-for-field match |
| Tokenizer, 44-case corpus | exact ids vs `RefTokenizer` **and** `sentencepiece` |
| Probe heads | contrastive **1.4e-6**, confidence **7.2e-5** |
| Batched + threaded prefill vs sequential | **bit-identical** |

**Needle v1** — 560 generated examples across five tool-name conventions, 0–8 parameters, 1–20 tools: **560/560 token-exact**.

Fixtures are committed, so the contract is version-pinned and reproducible without re-running Python. 341 Rust tests and 94 WASM binding assertions run in CI, in both the default and `parallel` feature configurations, alongside the C ABI and Python wheel.

A measured caution, because it cost two debugging sessions: the reference config ships `dtype="bfloat16"`, and measuring against a bfloat16 oracle makes a *correct* implementation look catastrophically wrong — a relative error of **9.7**, i.e. about 970%, not a small number with a missing exponent. Every figure above is measured against an f32 reference. See [docs/v3-port-record.md](docs/v3-port-record.md).

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="benchmarks">
  <img src="https://img.shields.io/badge/-Benchmarks-CE422B?style=flat-square" height="22" alt="Benchmarks"/>
</h2>

Apple M5 Max, steady state, threaded — which is what the CLI, Python and C crates build. **Needle v2** against the Python/JAX reference on the **same machine and the same weights**:

| | needle-rs | Python / JAX |
|---|---|---|
| Load model | **9 ms** | 761 ms + 26.5 s first-call JIT |
| Decode | **8.2 ms/token** | 11.0 ms/token |
| Prefill | 1.60 ms/token | **0.51 ms/token** |
| Session memory | **~23 MB** | — |
| Runtime dependencies | **0** | 4 |

Decode is **1.35× faster** and cold start about **50×** faster; prefill is slower, because the reference multiplies dense f32 weights while this runs from 2-bit packed ones. On a single query the two cross at **48 generated tokens** — faster above, slower below, and faster at any length on a cold process.

Prefill improved **3.6×** during the v2 port (5.74 → 1.60 ms/token) via batching, threading and batched Engram projections. The packed dot product is **2.9×** faster than a single-accumulator version and **9.1×** faster than a naive one.

**Needle v3**, same machine and flags, carries 121M parameters against v2's 45M and costs roughly 2–3× per token: 10 ms to load, **4.20 ms/token** prefill and **8.65 ms/token** decode. A 100-token prompt answers in about 700 ms, 424 ms of it to first token. Its `--kv-int8` cache costs about 2% of that speed and roughly a quarter of the memory. That is the trade v3 asks for, and the reason v2 is not deprecated.

Full methodology — including the optimisations that were measured and **rejected**, such as hand-written NEON losing to LLVM's autovectoriser — is in [BENCHMARKS.md](BENCHMARKS.md).

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="use-cases">
  <img src="https://img.shields.io/badge/-What_it's_good_for-CE422B?style=flat-square" height="22" alt="What it's good for"/>
</h2>

- **In-browser agents.** Route a user's sentence to one of your app's functions with no backend. See [`examples/browser-demo`](examples/browser-demo) and the [live demo](https://needle-rs.pages.dev).
- **Dynamic tool sets.** Generate tools from live state each turn and let the model pick — [`examples/dom-editor`](examples/dom-editor) rewrites a page from plain English.
- **Edge workers.** 537 KB of WASM fits inside a Cloudflare Worker.
- **Large tool catalogues.** Narrow hundreds of tools with the retrieval head before the call — Needle 2 only, which is the one generation that ships a contrastive head.
- **Uncertainty-aware routing.** Use the confidence head to escalate to a larger model only when needed.
- **Offline and embedded.** `no_std` kernels, one dependency, no allocator assumptions beyond `alloc`.

Not the right tool for open-ended chat, long-form generation, or reasoning beyond tool selection. It does one thing.

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="acknowledgements">
  <img src="https://img.shields.io/badge/-Acknowledgements-7D8590?style=flat-square" height="22" alt="Acknowledgements"/>
</h2>

Needle is designed and trained by [Henry Ndubuaku](https://github.com/hndubuaku) and the [Cactus Compute](https://github.com/cactus-compute) team. The model architecture, training code, dataset, and weights are entirely their work, released openly — the upstream repository under Apache-2.0, the Needle 3 weights under Apache-2.0, and the Needle 2 and Needle 1 weights under MIT. `needle-rs` is an independent Rust runtime — no upstream code is copied, only the published architecture is implemented. See [NOTICE](NOTICE).

**If you find this useful, please star the [upstream Needle repo](https://github.com/cactus-compute/needle) as well.**

<br/>

<!-- ────────────────────────────────────────────────── -->
<h2 id="citation">
  <img src="https://img.shields.io/badge/-Citation-7D8590?style=flat-square" height="22" alt="Citation"/>
</h2>

The model is Cactus Compute's work. Cite it as they ask — these are their
entries, reproduced verbatim from the upstream README. The design and ablations
are in the paper, [arXiv:2607.18363](https://arxiv.org/abs/2607.18363).

Current, and what to cite unless you mean an older generation specifically:

```bibtex
@misc{needle3_2026,
  title        = {Needle: Automation Foundation Model for Tiny Devices},
  author       = {Ndubuaku, Henry and Mosoyan, Karen and Mroz, Jakub and Cylich, Noah and
                  Kumar, Satyajit and Sandhu, Parkirat and Shemet, Roman and Lee, Justin H.},
  year         = {2026},
  organization = {Cactus Compute, Inc.},
  howpublished = {\url{https://github.com/cactus-compute/needle}}
}
```

If your work uses the v2 weights specifically:

```bibtex
@misc{needle2_2026,
  title        = {Needle 2: A 45M-Parameter Foundation Tool-Calling Model for Tiny Devices},
  author       = {Ndubuaku, Henry and Mosoyan, Karen and Mroz, Jakub and Cylich, Noah and
                  Kumar, Satyajit and Sandhu, Parkirat and Shemet, Roman and Lee, Justin H.},
  year         = {2026},
  organization = {Cactus Compute, Inc.},
  howpublished = {\url{https://github.com/cactus-compute/needle}}
}
```

If your work uses the v1 weights specifically, cite the v1 model instead:

```bibtex
@software{needle2026,
  author  = {Ndubuaku, Henry and {Cactus Compute}},
  title   = {Needle: A 26M-Parameter Tool-Calling Transformer},
  year    = {2026},
  url     = {https://github.com/cactus-compute/needle},
  license = {MIT}
}
```

And this runtime, if it is relevant to what you are reporting:

```bibtex
@software{needlers2026,
  author  = {Ibrahim, Abdalrahman},
  title   = {needle-rs: Pure-Rust WASM Runtime for Needle},
  year    = {2026},
  url     = {https://github.com/geekgineer/needle-rs},
  license = {MIT}
}
```

<br/>

<div align="center">
  <sub>needle-rs is MIT — see <a href="LICENSE">LICENSE</a> and <a href="NOTICE">NOTICE</a>. The models are by <a href="https://github.com/cactus-compute">Cactus Compute</a> under their own terms: Needle 3 Apache-2.0, Needle 2 and 1 MIT.</sub>
</div>