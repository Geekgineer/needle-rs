#!/usr/bin/env python3
"""Needle 3 through the Python bindings, against the real checkpoint.

The Rust tests prove the numerics. This proves the surface a Python caller
touches: the abstention contract crossing the FFI boundary as a value rather
than an exception, streaming deltas, and all three generations coexisting in
one module.

    VIRTUAL_ENV=.venv maturin develop --release -m crates/needle-python/Cargo.toml
    python crates/needle-python/tests/test_v3.py

Needs weights/needle3.cact and weights/needle2.cact.
"""

import os
import sys

if not os.path.exists("weights/needle3.cact"):
    print("skipping v3 python test: no weights/needle3.cact")
    sys.exit(0)

import json
from needle_rs import V3Engine, V2Engine

TOOLS = json.dumps([
    {"name": "get_weather", "description": "Get current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]}},
    {"name": "control_lights", "description": "Turn lights on or off in a room",
     "parameters": {"type": "object",
                    "properties": {"room": {"type": "string"}, "state": {"type": "string"}},
                    "required": ["room", "state"]}},
])
Q = "What's the weather in Paris?"

ok = fail = 0
def check(c, m):
    global ok, fail
    if c: ok += 1; print(f"  ok   {m}")
    else: fail += 1; print(f"  FAIL {m}")

e = V3Engine.load("weights/needle3.cact")
check(e is not None, "V3Engine loads")
check(e.max_seq_len == 8192, f"max_seq_len {e.max_seq_len}")

text = e.run(Q, TOOLS)
print("  run ->", repr(text))
check("get_weather" in text, "run produces a call")

payload = e.run_json(Q, TOOLS)
check(bool(payload) and payload.startswith("["), f"run_json -> {payload}")
check(V3Engine.reasoning(text) is not None, f"reasoning -> {V3Engine.reasoning(text)!r}")

d = e.generate(Q, TOOLS, constrain=True)
print("  generate keys:", sorted(d))
check(d["tool_call"] and "get_weather" in d["tool_call"], "constrained generate")
check(d["stop_reason"] in ("ImEnd", "Eos"), f"stop_reason {d['stop_reason']}")
check(isinstance(d["token_ids"], list) and len(d["token_ids"]) > 0, "token_ids present")

check(e.has_confidence(), "confidence head present")
p_right = e.confidence_for(Q, TOOLS, text)
p_bare = e.confidence_for(Q, TOOLS, "")
print(f"  confidence: completion {p_right:.4f}, bare query {p_bare:.4f}")
check(p_right > p_bare, "completion outscores a bare query")

res, p = e.run_scored(Q, TOOLS)
check(0.0 <= p <= 1.0, f"run_scored -> {p:.4f}")

pieces = []
streamed = e.run_stream(Q, TOOLS, pieces.append)
check("".join(pieces) == streamed, "streamed deltas reproduce the return value")

kv = e.kv_bytes(512)
print(f"  kv_bytes(512) = {kv/1024/1024:.1f} MB")
check(0 < kv < 20 * 1024**2, "kv_bytes is a sane figure")

kv8 = e.kv_bytes(512, kv_int8=True)
print(f"  kv_bytes(512, kv_int8=True) = {kv8/1024/1024:.1f} MB")
check(kv8 * 3 < kv, "the int8 cache figure is materially smaller")

# The quantised cache is the width the container declares; it must reach the
# same decision, not merely run.
free = e.generate(Q, TOOLS)
quant = e.generate(Q, TOOLS, kv_int8=True)
print(f"  int8 tool_call -> {quant['tool_call']}")
check(
    free["tool_call"] == quant["tool_call"],
    "the int8 cache produces the same tool call",
)

none = e.run_json("Write me a poem about the sea", TOOLS)
print("  poem ->", repr(none))
check(none in ("[]", None), "unrelated query abstains")

v2 = V2Engine.load("weights/needle2.cact")
check(v2 is not None, "V2Engine still loads from the same module")
try:
    V3Engine.load("weights/needle2.cact"); check(False, "a v2 container must be refused")
except Exception:
    check(True, "a v2 container is refused by V3Engine")

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
