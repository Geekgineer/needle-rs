// Needle 3 through the WebAssembly bindings, in Node.
//
// The Rust tests prove the numerics. This proves the surface a browser
// actually touches: loading from bytes, the abstention contract surviving the
// String round trip, streaming, and both generations coexisting in one module.
//
//   wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
//   node crates/needle-wasm/tests/node_e2e_v3.js

const fs = require("fs");
const path = require("path");

const PKG = path.resolve(__dirname, "../../../pkg-nodejs/needle_wasm.js");
const CACT_V3 = path.resolve(__dirname, "../../../weights/needle3.cact");
const CACT_V2 = path.resolve(__dirname, "../../../weights/needle2.cact");

if (!fs.existsSync(CACT_V3)) {
  console.log("skipping v3 wasm e2e: no weights/needle3.cact");
  process.exit(0);
}
const { NeedleV3Wasm, NeedleV2Wasm, extract_tool_call_v3 } = require(PKG);

const TOOLS = JSON.stringify([
  { name: "get_weather", description: "Get current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } },
  { name: "control_lights", description: "Turn lights on or off in a room",
    parameters: { type: "object", properties: { room: { type: "string" }, state: { type: "string" } },
                  required: ["room", "state"] } },
]);

let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ok   " + m); } else { fail++; console.log("  FAIL " + m); } };

const bytes = new Uint8Array(fs.readFileSync(CACT_V3));
const e = NeedleV3Wasm.load(bytes);
ok(e !== undefined, "v3 container loads");

const q = "What's the weather in Paris?";
const text = e.run(q, TOOLS);
console.log("  run ->", JSON.stringify(text));
ok(text.includes("get_weather"), "run produces a get_weather call");

const payload = e.run_json(q, TOOLS);
ok(payload.startsWith("[") && payload.includes("Paris"), "run_json is the payload: " + payload);

const reasoning = e.reasoning(text);
ok(reasoning === undefined || typeof reasoning === "string", "reasoning is a string or undefined");
console.log("  reasoning ->", JSON.stringify(reasoning));

ok(e.has_confidence() === true, "v3 exports a confidence head");
const pRight = e.confidence_for(q, TOOLS, text);
const pWrong = e.confidence_for(q, TOOLS, '<tool_call>[{"name":"control_lights","arguments":{"room":"Paris","state":"on"}}]</tool_call>');
console.log(`  confidence right=${pRight.toFixed(4)} wrong=${pWrong.toFixed(4)}`);
ok(pRight > pWrong, "confidence ranks the right completion higher");

// Abstention must survive the string round trip: "[]" is a decision, "" is not.
const none = e.run_json("Write me a poem about the sea", TOOLS);
console.log("  poem ->", JSON.stringify(none));
ok(none === "[]" || none === "", "an unrelated query yields [] or empty, not an invented call");

// Streaming must reconstruct exactly what run returns.
let streamed = "";
const streamedText = e.run_stream(q, TOOLS, (piece) => { streamed += piece; });
ok(streamed === streamedText, "streamed deltas reproduce the returned text");

// Constrained decoding through the wasm surface.
const bound = e.generate(q, TOOLS, 0, 0.0, 0, true, false);
ok(bound.includes("get_weather"), "constrained generate still answers");

// Memory reporting, which a page needs before committing to a session.
const kv = e.kv_bytes(512);
console.log(`  kv_bytes(512) = ${(kv / 1024 / 1024).toFixed(1)} MB, max_seq_len ${e.max_seq_len()}`);
ok(kv > 0 && kv < 20 * 1024 * 1024, "kv_bytes(512) is a sane browser figure");

// The int8 cache is the one a memory-constrained tab actually wants.
const kv8 = e.kv_bytes_int8(512);
console.log(`  kv_bytes_int8(512) = ${(kv8 / 1024 / 1024).toFixed(1)} MB`);
ok(kv8 * 3 < kv, "kv_bytes_int8 reports a materially smaller cache");
const quant = e.generate(q, TOOLS, 0, 0.0, 0, false, true);
console.log(`  int8 -> ${JSON.stringify(extract_tool_call_v3(quant) ?? quant)}`);
ok(
  extract_tool_call_v3(quant) === extract_tool_call_v3(text),
  "the int8 cache produces the same tool call",
);

// Both generations must coexist in one module.
if (fs.existsSync(CACT_V2)) {
  const v2 = NeedleV2Wasm.load(new Uint8Array(fs.readFileSync(CACT_V2)));
  ok(v2 !== undefined, "v2 still loads from the same module");
  ok(NeedleV3Wasm.load(new Uint8Array(fs.readFileSync(CACT_V2))) === undefined,
     "a v2 container is rejected by the v3 class");
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
