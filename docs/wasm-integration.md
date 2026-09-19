# WASM Integration Guide

How to load and use needle-rs in JavaScript environments — browsers, Node.js and
edge workers.

## Build the WASM package

```bash
# Web target (ES module, for browsers and bundlers)
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/

# Node.js target (CommonJS, for server-side or testing)
wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
```

Output: `pkg/needle_wasm_bg.wasm` (633 KB) + `pkg/needle_wasm.js` (40 KB of glue).

`wasm-pack` does **not** run `wasm-opt` here — the crate sets
`wasm-opt = false`, because the binary wasm-pack downloads fails in this build
environment. Run it yourself to get the published 560 KB module:

```bash
wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
  pkg/needle_wasm_bg.wasm -o pkg/needle_wasm_bg.wasm
```

That is 221 KB gzipped and 203 KB as Cloudflare Pages serves it (brotli);
`brotli -q 11` gets it to 131 KB. Serve it compressed — it is the single
biggest win available, larger than anything `-Oz` does.

## One module, three model generations

| | Needle v3 | Needle v2 | Needle v1 |
|---|---|---|---|
| Class | `NeedleV3Wasm` | `NeedleV2Wasm` | `NeedleWasm` |
| `load` | `load(cactBytes)` | `load(cactBytes)` | `load(weightsBytes, vocabText)` |
| Assets | one `.cact` file (35.3 MB) | one `.cact` file (13.7 MB) | `.safetensors` + vocab text |
| Extra methods | `run_json`, `generate`, `reasoning`, `confidence_for`, `kv_bytes` | `run_json`, `generate`, `confidence_for` | `run_batch` |
| Probe heads | confidence | contrastive + confidence | contrastive |
| Context | 8192 | 2048 | 1024 |

All three classes are exported from the same module and can be instantiated side
by side. Pick per deployment; nothing forces a build-time choice.

v2 and v3 both use `.cact`. A container states its generation in its first word,
so `NeedleV3Wasm.load` on a v2 container returns `undefined` rather than
misreading the header.

---

## Needle v3 in a web page (no bundler)

```html
<script type="module">
  import init, { NeedleV3Wasm } from "./pkg/needle_wasm.js";

  await init();
  const cact = new Uint8Array(
    await (await fetch("./needle3.cact")).arrayBuffer()
  );
  const model = NeedleV3Wasm.load(cact);
  if (!model) throw new Error("not a Needle 3 container");

  const tools = JSON.stringify([{
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  }]);

  // v3 reasons first, so `run` returns a <think> block then the call.
  const text = model.run("What's the weather in Paris?", tools);
  console.log(model.reasoning(text));   // the chain-of-thought, or undefined
  console.log(model.run_json("What's the weather in Paris?", tools));
  // -> [{"name":"get_weather","arguments":{"city":"Paris"}}]
</script>
```

`run_json` returns `"[]"` when the model considered the tools and declined, and
`""` when it emitted no `<tool_call>` markers at all. Treating the first as a
failure turns a considered "no" into an error.

### Budgeting memory in a tab

A browser tab has a memory budget and v3's container is 35 MB, so the cache cost
is worth asking about before committing to a session length:

```js
model.kv_bytes(512);        //  8.8 MB  — f32 cache
model.kv_bytes_int8(512);   //  2.3 MB  — 8-bit cache
model.max_seq_len();        //  8192
```

Pass `kv_int8 = true` as the last argument to `generate` to use the 8-bit cache.
It is the width the container declares, costs about 2% of the speed, and on our
tests produces identical tool calls — but it is not bit-identical to the
default, so it stays opt-in:

```js
const out = model.generate(query, tools, 0, 0.0, 0, /*constrain*/ false,
                           /*kv_int8*/ true);
```

### Confidence scores the completion, not the query

```js
const completion = model.run(query, tools);
const p = model.confidence_for(query, tools, completion);
```

Pass the **completion**. The head scores a finished judgement: a correct call
scores 0.94, a wrong one 0.26, and a bare query 0.80 — which looks like a
confident answer and is not one. v2's head collapsed to near zero on a bare
query, so the misuse announced itself; v3's does not.

There is no `retrieve_tools` on `NeedleV3Wasm`: v3 exports a confidence head and
nothing else, so the method is absent rather than present-and-always-empty.

---

## Needle v2 in a web page (no bundler)

```html
<script type="module">
  import init, { NeedleV2Wasm } from "./pkg/needle_wasm.js";
  await init();

  // One request: the container carries weights, geometry and tokenizer.
  const res = await fetch("https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact");
  const engine = NeedleV2Wasm.load(new Uint8Array(await res.arrayBuffer()));

  const tools = JSON.stringify([{
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  }]);

  const query = "What's the weather in Paris?";
  const out = engine.run(query, tools);
  // <tool_call>[{"name":"get_weather","arguments":{"city":"Paris"}}]</tool_call>

  const payload = engine.run_json(query, tools);
  // [{"name":"get_weather","arguments":{"city":"Paris"}}]
  // "[]" when no tool applies; "" only if no markers were emitted at all

  // Gate on the model's own confidence in that answer.
  const p = engine.confidence_for(query, tools, out);
  if (p !== null && p >= 0.5) console.log(JSON.parse(payload));
  else console.warn(`low confidence (${p}) — escalate instead`);
</script>
```

Two "no call" cases, and they mean different things. An off-topic or
unsupportable query returns the string `"[]"` — the model deliberately emitted
an empty call, and that is the abstention you should act on. An **empty string**
means no `<tool_call>` markers appeared at all, which is a degenerate
generation. So parse it, and treat an empty array and an empty string
differently:

```js
const payload = engine.run_json(query, tools);
if (!payload) { /* no markers — retry or escalate */ }
else if (JSON.parse(payload).length === 0) { /* no tool applies */ }
else { /* act on the call */ }
```

(The C ABI signals the second case with a NULL return; the JS surface uses `""`
because wasm-bindgen maps a plain `String`.)

Tool schemas are compacted internally, so a pretty-printed `tools` string gives
byte-identical output to its minified form — verified, not assumed.

### Constrained and sampled generation

```js
// generate(query, tools, maxNewTokens, temperature, seed, constrain)
engine.generate(query, tools, 96, 0.0, 0, true);   // greedy, schema-constrained
engine.generate(query, tools, 96, 0.8, 42, false); // sampled, seed-reproducible
```

`constrain` restricts the payload to the declared schema — valid tool names and
argument keys only. It guarantees syntactic validity, not semantic correctness.
Constrained output is not streamable: the grammar mask depends on the whole
payload so far.

### Streaming tokens

```js
const text = engine.run_stream(query, tools, (tokenId, piece) => {
  outputEl.append(piece);   // or process.stdout.write(piece) in Node
});
// `text` is the full output. For v2 it equals the concatenated pieces.
```

### Probe heads

```js
engine.confidence_for(query, tools, out);  // probability in (0,1), or null
engine.confidence(text);                   // raw logit — see the caveat below
engine.contrastive_dim();                  // 128, or 0 without the head
engine.encode_contrastive("weather forecast");  // Float32Array, L2-normalised
engine.retrieve_tools(query, JSON.stringify(descs), 3);
// '[[0,0.86],[1,0.45],[2,0.31]]' — [index, score] pairs, descending
```

The confidence head scores a *judgement already made*: it is trained on the
formatted prompt followed by a completion. `confidence_for` assembles that input
for you. Passing a bare query to the lower-level `confidence` reads near zero
however answerable the query is — that is the primitive behaving as intended,
not a bug. Upstream's published score additionally takes the minimum with the
decode probability of the call tokens; needle-rs exposes the head only, because
the reference implementation of that composition is not published.

Embeddings are L2-normalised on both sides, so cosine similarity is a plain dot
product.

---

## Needle v1 in a web page

```html
<script type="module">
  import init, { NeedleWasm } from "./pkg/needle_wasm.js";
  await init();

  const HF = "https://huggingface.co/Abdalrahman/needle-rs-safetensors/resolve/main";
  const [weights, vocab] = await Promise.all([
    fetch(`${HF}/needle.safetensors`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
    fetch(`${HF}/vocab.txt`).then(r => r.text()),
  ]);

  const engine = NeedleWasm.load(weights, vocab);
  const result = engine.run("Book a flight from London to JFK tomorrow", toolsJson);
  // [{"name":"book_flight","arguments":{"origin":"London","destination":"JFK","date":"tomorrow"}}]
</script>
```

`NeedleWasm.load` returns `undefined` on failure rather than throwing — check it.

v1 post-processes its output: the `<tool_call>` marker is stripped and the
caller's original tool-name casing is restored. So with `run_stream`, the
streamed pieces are a progress view and the **returned** string is the answer —
a streamed `get_weather` can come back as `getWeather`. Render the pieces as
they arrive, then settle on the return value.

v1 also has batch inference, which v2 does not expose in WASM:

```js
const results = engine.run_batch([
  { query: "What is the weather in Paris?", tools },
  { query: "Email bob@example.com about lunch", tools },
]);
// a JS Array of strings
```

---

## In Node.js

```js
const { NeedleV2Wasm } = require("./pkg-nodejs/needle_wasm.js");
const fs = require("fs");

const engine = NeedleV2Wasm.load(fs.readFileSync("weights/needle2.cact"));
console.log(engine.run("What's the weather in Tokyo?", tools));
```

`crates/needle-wasm/tests/node_e2e_v2.js` is a runnable end-to-end suite over
both classes — 36 assertions. Build the `nodejs` target first, then:

```bash
node crates/needle-wasm/tests/node_e2e_v2.js
```

---

## In a Cloudflare Worker

```js
import init, { NeedleV2Wasm } from "./pkg/needle_wasm.js";
import wasmModule from "./pkg/needle_wasm_bg.wasm";

let engine;

export default {
  async fetch(request, env) {
    if (!engine) {
      await init(wasmModule);
      // Serve the container from R2 in production: a cross-origin fetch of
      // 13.7 MB on a cold start will dominate your request budget.
      const res = await fetch(env.CACT_URL);
      engine = NeedleV2Wasm.load(new Uint8Array(await res.arrayBuffer()));
    }
    const { query, tools } = await request.json();
    return new Response(engine.run_json(query, tools) || "[]", {
      headers: { "Content-Type": "application/json" },
    });
  },
};
```

`engine` survives between requests on a warm isolate but not across evictions,
so treat the load as recoverable and keep it inside the guard.

---

## Memory notes

- A v2 engine needs roughly 23 MB of WASM linear memory: the container's
  weights plus a KV cache sized to the 256-token attention window, not to
  `max_seq_len`. See the KV-ring note in [ARCHITECTURE.md](../ARCHITECTURE.md).
- A v3 engine is heavier: 35.3 MB of container plus a cache that grows with the
  session — 8.8 MB at 512 tokens and 42.0 MB at the full 8192, or 2.3 MB and
  11.2 MB with `kv_int8`. Ask `kv_bytes`/`kv_bytes_int8` rather than guessing;
  v3's four global layers hold the whole context while its sixteen local layers
  hold a 1024-token window, so the cost is not linear in the obvious way.
- WASM linear memory never shrinks. Plan for one engine per tab or per isolate,
  and reuse the handle rather than reloading.
- All three classes run single-threaded in WASM. The `parallel` feature is native
  only; without `SharedArrayBuffer` there is nothing to parallelise onto, and
  that is the expected case in a browser.
- wasm32 addresses 4 GB, so headroom is not the constraint — download time is.
