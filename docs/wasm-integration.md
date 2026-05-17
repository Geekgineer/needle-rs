# WASM Integration Guide

How to load and use needle-rs in JavaScript environments.

## Build the WASM package

```bash
# Web target (ES module, for browsers and bundlers)
wasm-pack build crates/needle-wasm --target web --release --out-dir ../../pkg/

# Node.js target (CommonJS, for server-side or testing)
wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg/
```

Output: `pkg/needle_wasm_bg.wasm` (260 KB, `wasm-opt -Oz` applied) + `pkg/needle_wasm.js` (JS glue).

---

## In a web page (no bundler)

```html
<script type="module">
  import init, { NeedleWasm } from "./pkg/needle_wasm.js";
  await init(); // initialize WASM

  // Load weights + vocab (fetch from wherever — HuggingFace, CDN, local)
  const weightsResp = await fetch("https://huggingface.co/abdalrahman/needle-rs-safetensors/resolve/main/needle.safetensors");
  const weightsBytes = new Uint8Array(await weightsResp.arrayBuffer());

  const vocabResp = await fetch("https://huggingface.co/abdalrahman/needle-rs-safetensors/resolve/main/vocab.txt");
  const vocabText = await vocabResp.text();

  // Load the engine (once; cache this handle)
  const engine = NeedleWasm.load(weightsBytes, vocabText);

  // Run inference
  const tools = JSON.stringify([{
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: {
      type: "object",
      properties: { location: { type: "string" } }
    }
  }]);

  const result = engine.run("What is the weather in Paris?", tools);
  console.log(result); // {"name":"get_weather","arguments":{"location":"Paris"}}
</script>
```

---

## Streaming tokens

```js
const result = engine.run_stream(
  "Book a flight from London to New York",
  tools,
  (tokenId, piece) => {
    process.stdout.write(piece); // or update DOM
  }
);
// result is the final post-processed string (same as run())
```

---

## Batch inference

```js
const results = engine.run_batch([
  { query: "What is the weather in Paris?", tools },
  { query: "Translate 'hello' to Spanish", tools: translatorTools },
]);
// results is a JS Array of strings
```

---

## Contrastive retrieval

```js
// Get an embedding for semantic search
const embedding = engine.encode_contrastive("weather forecast");
// embedding is Float32Array (L2-normalized) or null if no contrastive head

// Rank tools by relevance to a query
const topK = engine.retrieve_tools(
  "What is the weather in Paris?",
  JSON.stringify(["Get weather data", "Search the web", "Send an email"]),
  2  // top_k
);
// topK: '[{"index":0,"score":0.95},{"index":1,"score":0.42}]'
```

---

## In Node.js

```js
const { NeedleWasm } = require("./pkg/needle_wasm.js");
const fs = require("fs");

const weightsBytes = new Uint8Array(fs.readFileSync("weights/needle.safetensors").buffer);
const vocabText = fs.readFileSync("weights/vocab.txt", "utf8");

const engine = NeedleWasm.load(weightsBytes, vocabText);
const result = engine.run("What is the weather in Tokyo?", tools);
console.log(result);
```

---

## In a Cloudflare Worker

```js
// worker.js
import init, { NeedleWasm } from "./pkg/needle_wasm.js";
import wasmBytes from "./pkg/needle_wasm_bg.wasm";

let engine;

export default {
  async fetch(request, env) {
    if (!engine) {
      await init(wasmBytes);
      // Fetch weights from R2 or HuggingFace on first request
      const resp = await fetch(env.WEIGHTS_URL);
      const weightsBytes = new Uint8Array(await resp.arrayBuffer());
      const vocabResp = await fetch(env.VOCAB_URL);
      const vocabText = await vocabResp.text();
      engine = NeedleWasm.load(weightsBytes, vocabText);
    }

    const { query, tools } = await request.json();
    const result = engine.run(query, tools);
    return new Response(result, { headers: { "Content-Type": "application/json" } });
  }
};
```

---

## Memory notes

- `NeedleWasm.load()` allocates ~50 MB of WASM linear memory (weights + KV caches).
- The WASM linear memory cannot be freed mid-session; plan for one engine instance per tab.
- On browsers without `SharedArrayBuffer`, multi-threading is unavailable — this is the expected case and inference runs single-threaded.
- WASM memory limit is 4 GB on wasm32; the 23 MB total footprint leaves ample headroom.
