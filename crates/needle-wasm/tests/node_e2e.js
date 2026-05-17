/**
 * WASM end-to-end test for needle-wasm (Node.js).
 *
 * Prerequisites:
 *   wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
 *
 * Run from workspace root:
 *   node crates/needle-wasm/tests/node_e2e.js
 *
 * Exit code 0 = pass, non-zero = fail.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const PKG = path.resolve(__dirname, "../../../pkg-nodejs/needle_wasm.js");
const WEIGHTS = path.resolve(__dirname, "../../../weights/needle.safetensors");
const VOCAB = path.resolve(__dirname, "../../../weights/vocab.txt");

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    failed++;
  } else {
    console.log(`PASS: ${msg}`);
    passed++;
  }
}

async function main() {
  // ── Prerequisites ────────────────────────────────────────────────────────
  if (!fs.existsSync(PKG)) {
    console.error(`SKIP: ${PKG} not found — run wasm-pack build first`);
    process.exit(0);
  }
  if (!fs.existsSync(WEIGHTS) || !fs.existsSync(VOCAB)) {
    console.error(`SKIP: weights not found at ${WEIGHTS}`);
    process.exit(0);
  }

  const { NeedleWasm } = require(PKG);

  const weightsBytes = fs.readFileSync(WEIGHTS);
  const vocabText = fs.readFileSync(VOCAB, "utf8");

  // ── Load ─────────────────────────────────────────────────────────────────
  const engine = NeedleWasm.load(new Uint8Array(weightsBytes), vocabText);
  assert(engine !== null && engine !== undefined, "NeedleWasm.load() returns non-null");

  const query = "What's the weather in Paris?";
  const tools = JSON.stringify([
    {
      name: "get_weather",
      description: "Get weather",
      parameters: { location: { type: "string" } },
    },
  ]);

  // ── run() ────────────────────────────────────────────────────────────────
  const result = engine.run(query, tools);
  assert(typeof result === "string", "run() returns a string");
  assert(!result.startsWith("<tool_call>"), "run() strips <tool_call> prefix");
  assert(result.includes("get_weather"), "run() output contains tool name");
  assert(result.includes("location"), "run() output contains arg key");
  console.log(`  run() output: ${result}`);

  // ── Output is valid JSON ──────────────────────────────────────────────────
  let parsed;
  try {
    parsed = JSON.parse(result.trim());
    assert(Array.isArray(parsed), "run() output is a JSON array");
    assert(parsed.length > 0, "run() output array is non-empty");
    assert(parsed[0].name === "get_weather", "run() output has correct tool name");
  } catch (e) {
    assert(false, `run() output is not valid JSON: ${e.message}`);
  }

  // ── run_stream() ─────────────────────────────────────────────────────────
  let tokenCount = 0;
  const streamResult = engine.run_stream(query, tools, (tokenId, piece) => {
    tokenCount++;
    assert(typeof tokenId === "number", `stream token_id is number (got ${typeof tokenId})`);
    assert(typeof piece === "string", `stream piece is string (got ${typeof piece})`);
  });
  assert(tokenCount > 0, `run_stream() callback fired ${tokenCount} times (> 0)`);
  assert(streamResult === result, "run_stream() output matches run()");
  console.log(`  run_stream() callback fired ${tokenCount} times`);

  // ── run_batch() ──────────────────────────────────────────────────────────
  const examples = [
    { query, tools },
    {
      query: "Search for Python tutorials",
      tools: JSON.stringify([
        { name: "web_search", description: "Search", parameters: { query: { type: "string" } } },
      ]),
    },
  ];
  const batchResults = engine.run_batch(examples);
  assert(Array.isArray(batchResults), "run_batch() returns an Array");
  assert(batchResults.length === examples.length, `run_batch() returns ${examples.length} results`);
  assert(batchResults[0] === result, "run_batch()[0] matches run() for same input");
  console.log(`  run_batch() results: [${batchResults.map((r) => JSON.stringify(r)).join(", ")}]`);

  // ── contrastive_dim() and encode_contrastive() ────────────────────────────
  const dim = engine.contrastive_dim();
  assert(typeof dim === "number", `contrastive_dim() returns number (got ${dim})`);
  console.log(`  contrastive_dim: ${dim}`);

  if (dim > 0) {
    const emb = engine.encode_contrastive(query);
    assert(emb !== null && emb !== undefined, "encode_contrastive() returns non-null");
    assert(emb instanceof Float32Array, "encode_contrastive() returns Float32Array");
    assert(emb.length === dim, `encode_contrastive() length equals contrastive_dim (${dim})`);

    let sqNorm = 0;
    for (let i = 0; i < emb.length; i++) sqNorm += emb[i] * emb[i];
    assert(Math.abs(sqNorm - 1.0) < 1e-4, `encode_contrastive() is L2-normalized (||v||²=${sqNorm.toFixed(6)})`);

    // ── retrieve_tools() ───────────────────────────────────────────────────
    const descs = [
      "Get current weather conditions for a location",
      "Search the web for information",
      "Send an email to a recipient",
    ];
    const rankJson = engine.retrieve_tools(query, JSON.stringify(descs), 2);
    assert(typeof rankJson === "string", "retrieve_tools() returns a string");
    let ranks;
    try {
      ranks = JSON.parse(rankJson);
      assert(Array.isArray(ranks), "retrieve_tools() output is a JSON array");
      assert(ranks.length === 2, `retrieve_tools(top_k=2) returns 2 results, got ${ranks.length}`);
      assert(
        ranks[0].score >= ranks[1].score,
        `retrieve_tools() sorted by descending score (${ranks[0].score} >= ${ranks[1].score})`
      );
      console.log(`  retrieve_tools() top-2: ${rankJson}`);
    } catch (e) {
      assert(false, `retrieve_tools() output is not valid JSON: ${e.message}`);
    }
  } else {
    console.log("  SKIP contrastive/retrieval tests: no contrastive head in these weights");
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Uncaught error:", err);
  process.exit(1);
});
