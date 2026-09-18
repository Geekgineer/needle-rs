# Browser demo

A single self-contained HTML page that runs Needle inference entirely
client-side — **all three model generations, side by side**. No server, no
bundler, no API key. This is the source for the deployed demo at
**[needle-rs.pages.dev](https://needle-rs.pages.dev)**.

Pick the generation with the model cards at the top. The page reads the loaded
version's capabilities and enables only the controls that apply, so switching
between them is not a different code path — it is the same UI over three
engines.

## What it demonstrates

- **One file to load (v3 and v2).** A `.cact` container carries the weights, the
  architecture geometry *and* the tokenizer, so the page fetches 35.3 MB for v3
  or 13.7 MB for v2 and nothing else — no vocabulary, no config side-car. v1
  needs two fetches: 22 MB of weights plus a vocabulary.
- **Tool calling**, streamed token by token, with the `<think>` reasoning trace
  shown separately from the parsed call. v3 reasons on essentially every query;
  v2 does so sometimes (2 of 3 sample prompts). v1 emits no reasoning trace and post-processes its
  output — so the demo renders streamed pieces as they arrive, then settles on
  the returned string, which is the answer.
- **Grammar-constrained decoding** (v3 and v2), toggleable, so the payload
  cannot name an undeclared tool or argument key. v1 is always constrained and
  greedy-only.
- **The probe heads**, which differ by generation and are the clearest example
  of why the UI is capability-driven rather than version-branched:

  | | v3 | v2 | v1 |
  |---|---|---|---|
  | Confidence scoring | ✓ | ✓ | — |
  | Contrastive tool retrieval | — | ✓ | ✓ |

  v3 exports a confidence head and nothing else, so the retrieval panel is
  hidden rather than shown empty.

The spec cards read measured values: the runtime size comes from the actual
`PerformanceResourceTiming` entry for the wasm module and the model size from the
bytes received, so they cannot drift from reality.

## Running locally

Build the WASM package, then serve:

```bash
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/
./examples/browser-demo/serve.sh
# → http://localhost:8080
```

`serve.sh` copies `pkg/` next to `index.html` in a temp directory, mirroring the
CI deployment layout. The model streams from HuggingFace, so no local weights are
needed.

## Verifying the bindings

The page only calls methods that are covered by a test:

```bash
wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
node crates/needle-wasm/tests/node_e2e.js      # v1, 44 assertions
node crates/needle-wasm/tests/node_e2e_v2.js   # v2, 36 assertions
node crates/needle-wasm/tests/node_e2e_v3.js   # v3, 14 assertions
```

Between them those exercise `load`, `run`, `run_json`, `generate` (greedy,
constrained, sampled and with the int8 cache), `run_stream`, `reasoning`,
`confidence_for`, `encode_contrastive`, `retrieve_tools`, `kv_bytes` and
`kv_bytes_int8` against the real weights — 94 assertions in total, and all three
run in CI.

## Notes

- The probe heads attend over the whole sequence rather than a sliding window, so
  they allocate a larger KV cache than generation does. That is why the demo puts
  them behind an **Analyse** button instead of running them on every query.
- Constrained decoding is not streamed: the grammar mask depends on the whole
  payload emitted so far, so that path returns the finished text.
- The confidence head scores a completed judgement, not a question, so
  **Analyse** scores the prompt together with the model's own output via
  `confidence_for`. Passing the bare query instead reads near zero on v2 — and,
  more dangerously, about 0.80 on v3, which looks like a confident answer and is
  not one.
- A v3 session costs noticeably more memory than a v2 one: 35.3 MB of container
  plus a key/value cache that grows with the conversation. `kv_bytes(seq_len)`
  and `kv_bytes_int8(seq_len)` report it, and a tab on a memory budget should ask
  before committing to a long session.

## Deployment

`.github/workflows/wasm-demo.yml` builds and deploys this to Cloudflare Pages on
every push to `main`, and `release.yml` does the same on a tag. Both need
`CF_API_TOKEN` and `CF_ACCOUNT_ID` as repository secrets — see
[docs/RELEASING.md](../../docs/RELEASING.md).
