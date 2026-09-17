#!/usr/bin/env python3
"""The canon check for the Needle 3 container.

Rebuilds the Flax parameter tree from `needle3.cact` and runs upstream's own
model on it. Per-tensor parity cannot catch a canon error — the directory is
nameless, so a correct tensor in the wrong positional slot still passes every
individual comparison. Coherent tool calls out of this script mean the
ordering, the mixed-width dequantisation and the tokenizer are all right end
to end, and it is the oracle the Rust forward pass gets verified against.

    JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
        tools/check_v3_canon.py

Slow by design: upstream's `generate` recomputes the whole buffer each step
rather than caching, so this is O(n^2) over 121M parameters on CPU.
"""

import sys
import time

import jax.numpy as jnp
import numpy as np

from cact_params_v3 import rebuild
from needle.model.architecture import SimpleAttentionNetwork
from needle.model.run import build_prompt
from needle.model.tokenizer import BOS_ID, EOS_ID, PAD_ID, get_tokenizer

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "control_lights",
        "description": "Turn lights on or off in a room",
        "parameters": {
            "type": "object",
            "properties": {"room": {"type": "string"}, "state": {"type": "string"}},
            "required": ["room", "state"],
        },
    },
]

QUERIES = [
    "What's the weather in Paris?",
    "Turn off the bedroom lights",
]

MAX_NEW = 40


def main():
    t0 = time.time()
    params, cfg, meta = rebuild()
    lm_params = {k: v for k, v in params.items() if not k.startswith("_")}
    print(f"rebuilt params in {time.time() - t0:.1f}s: "
          f"{cfg.num_layers}x{cfg.d_model}, {meta['num_tensors']} tensors")

    model = SimpleAttentionNetwork(cfg)
    tok = get_tokenizer(cfg.vocab_size)

    ok = True
    for query in QUERIES:
        prompt = build_prompt(query, TOOLS)
        ids = [BOS_ID] + tok.encode(prompt)
        buf_len = len(ids) + MAX_NEW
        buf = jnp.full((1, buf_len), PAD_ID, dtype=jnp.int32)
        buf = buf.at[0, : len(ids)].set(jnp.array(ids, dtype=jnp.int32))

        t1 = time.time()
        out = []
        for pos in range(len(ids) - 1, buf_len - 1):
            logits = model.apply({"params": lm_params}, buf)[0, pos]
            nxt = int(jnp.argmax(logits))
            if nxt == EOS_ID:
                break
            out.append(nxt)
            buf = buf.at[0, pos + 1].set(nxt)

        text = tok.decode(out)
        print(f"\nQ: {query}")
        print(f"A: {text}")
        print(f"   ({len(out)} tokens, {time.time() - t1:.1f}s, "
              f"prompt {len(ids)} tokens)")
        if "tool_call" not in text and "name" not in text:
            ok = False

    print()
    if ok:
        print("CANON OK — the rebuilt tree produces coherent tool calls, so the "
              "positional ordering and mixed-width dequantisation are right.")
    else:
        print("CANON SUSPECT — output is not tool-call shaped. Do not build the "
              "Rust forward pass on this ordering until it is.")
        sys.exit(1)


if __name__ == "__main__":
    main()
