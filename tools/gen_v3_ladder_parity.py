#!/usr/bin/env python3
"""Per-rung forward parity for the Needle 3 intelligence ladder.

Needle 3 is laddered: every depth from 2 to 20 blocks is a trained subnetwork.
`V3Engine::load_with_depth` slices one out of the full container at load time,
and this records what upstream's own slicer produces from the *same* container
so the two can be compared.

Both sides therefore start from `needle3.cact` — identical weights, identical
quantisation. That is the point: a container built by `needle build --layers N`
is a different quantisation of the same model (the published archive is 2-bit
for most tensors while the public exporter emits 4), so comparing against one
would confound a slicing error with a requantisation difference. Slicing the
same bytes on both sides isolates the slicer.

    JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
        tools/gen_v3_ladder_parity.py [depths]

Emits `tests/v3_ladder_vectors.json` + `.f32`, both gitignored for size.
"""

import json
import pathlib
import struct
import sys
import time

import jax.numpy as jnp
import numpy as np

from cact_params_v3 import rebuild
from needle.model.architecture import SimpleAttentionNetwork
from needle.model.finetune import rung
from needle.model.run import build_prompt
from needle.model.tokenizer import BOS_ID, get_tokenizer

OUT_JSON = pathlib.Path("tests/v3_ladder_vectors.json")
OUT_F32 = pathlib.Path("tests/v3_ladder_vectors.f32")

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]
QUERY = "What's the weather in Paris?"


class Blob:
    def __init__(self):
        self.buf = bytearray()

    def put(self, arr):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        off = len(self.buf) // 4
        self.buf += a.tobytes()
        return {"off": off, "len": int(a.size)}


def main():
    depths = [int(x) for x in sys.argv[1:]] or [2, 4, 6, 8, 12, 16, 20]
    t0 = time.time()
    base_params, base_cfg, _ = rebuild()
    print(f"rebuilt the 20-block parent in {time.time() - t0:.1f}s")

    tok = get_tokenizer(base_cfg.vocab_size)
    prompt = build_prompt(QUERY, TOOLS)
    ids = [BOS_ID] + tok.encode(prompt)
    tokens = jnp.array([ids], dtype=jnp.int32)
    print(f"prompt {len(ids)} tokens")

    blob = Blob()
    rungs = []
    for depth in depths:
        t = time.time()
        # Upstream's own slicer, on the same weights our loader reads.
        params, cfg = rung(base_params, base_cfg, depth)
        lm = {k: v for k, v in params.items() if not k.startswith("_")}
        # float32 for the same reason as the 20-block ladder: a bfloat16 oracle
        # cannot separate a wrong implementation from bf16's three decimals.
        cfg.dtype = "float32"
        model = SimpleAttentionNetwork(cfg)
        logits = np.asarray(model.apply({"params": lm}, tokens), dtype=np.float32)

        rec = {
            "depth": depth,
            "num_layers": int(cfg.num_layers),
            "global_layers": [int(x) for x in cfg.global_layers],
            "engram_layers": [int(x) for x in cfg.engram_layers],
            "logits": blob.put(logits),
            "rows": int(logits.shape[-1]),
            "argmax": [int(i) for i in np.argmax(logits[0], axis=-1)],
        }
        rungs.append(rec)
        print(
            f"  depth {depth:>2}: global={rec['global_layers']} "
            f"engram={rec['engram_layers']} logits{logits.shape} "
            f"[{time.time() - t:.1f}s]"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "source": "needle3.cact sliced by upstream needle.model.finetune.rung",
                "dtype": "float32",
                "query": QUERY,
                "tools": TOOLS,
                "tokens": ids,
                "parent_layers": int(base_cfg.num_layers),
                "rungs": rungs,
            },
            indent=1,
        )
    )
    OUT_F32.write_bytes(bytes(blob.buf))
    print(f"wrote {OUT_JSON} and {OUT_F32} ({len(blob.buf) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
