#!/usr/bin/env python3
"""Capture a Needle 3 forward-pass ladder from upstream's model.

Rebuilds the parameter tree from `needle3.cact` (see `cact_params_v3.py`,
whose ordering is pinned by `check_v3_canon.py`) and records intermediates at
every stage, so a Rust mismatch localises to one component instead of "the
logits are wrong".

Floats go to a side-car `.f32` so the JSON stays readable; the JSON holds
offsets into it.

    JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
        tools/gen_v3_forward_parity.py

Emits `tests/v3_forward_vectors.json` + `.f32`, both gitignored for size —
regenerate locally, as with the v2 ladder.
"""

import json
import pathlib
import struct
import sys
import time

import jax.numpy as jnp
import numpy as np

from cact_params_v3 import rebuild
from needle.model.architecture import (
    SimpleAttentionNetwork,
    engram_geometry,
    head_dims,
    make_causal_mask,
)
from needle.model.run import build_prompt
from needle.model.tokenizer import BOS_ID, get_tokenizer

OUT_JSON = pathlib.Path("tests/v3_forward_vectors.json")
OUT_F32 = pathlib.Path("tests/v3_forward_vectors.f32")

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
    """Append-only f32 side-car; returns (offset, shape) for each array."""

    def __init__(self):
        self.buf = bytearray()

    def add(self, arr):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        off = len(self.buf) // 4
        self.buf.extend(a.tobytes())
        return {"offset": off, "len": int(a.size), "shape": list(np.asarray(arr).shape)}


def main():
    t0 = time.time()
    params, cfg, meta = rebuild()
    lm = {k: v for k, v in params.items() if not k.startswith("_")}
    # float32 deliberately. The config ships bfloat16, which is what upstream
    # runs, but a bf16 oracle cannot separate "the Rust is wrong" from "bf16
    # has three decimal digits" — against it a correct forward reads as 8e-2
    # relative on the logits. Numeric parity is measured in f32; behavioural
    # parity is the separate token-exact end-to-end gate.
    cfg.dtype = "float32"
    model = SimpleAttentionNetwork(cfg)
    tok = get_tokenizer(cfg.vocab_size)
    print(f"rebuilt in {time.time() - t0:.1f}s")

    prompt = build_prompt(QUERY, TOOLS)
    ids = [BOS_ID] + tok.encode(prompt)
    tokens = jnp.array([ids], dtype=jnp.int32)
    n = len(ids)
    print(f"prompt {n} tokens")

    blob = Blob()
    rec = {
        "tokens": ids,
        "geometry": {
            "d_model": cfg.d_model,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "qk_head_dim": head_dims(cfg)[0],
            "v_head_dim": head_dims(cfg)[1],
            "sliding_window": int(getattr(cfg, "sliding_window", 0)),
            "global_layers": list(getattr(cfg, "global_layers", ())),
            "qkv_conv_taps": int(getattr(cfg, "qkv_conv_taps", 0)),
            "engram_layers": list(cfg.engram_layers),
            "out_vocab": int(getattr(cfg, "out_vocab", 0) or cfg.vocab_size),
            "rope_theta": float(cfg.rope_theta),
        },
        "dtype": "float32",
        "stages": {},
    }

    # Stage 1 — scaled input embeddings.
    t1 = time.time()
    emb = model.apply({"params": lm}, tokens, method=SimpleAttentionNetwork._input_embeddings)
    rec["stages"]["input_embeddings"] = blob.add(emb[0])
    print(f"  input_embeddings {tuple(emb.shape)}  {time.time() - t1:.1f}s")

    # Stage 2 — RoPE tables, which depend only on qk_head_dim and theta.
    rope = model.apply({"params": lm}, n, method=SimpleAttentionNetwork._rope)
    cos, sin = rope if isinstance(rope, (tuple, list)) else (rope[0], rope[1])
    rec["stages"]["rope_cos"] = blob.add(cos)
    rec["stages"]["rope_sin"] = blob.add(sin)
    print(f"  rope {np.asarray(cos).shape}")

    # Stage 3 — Engram keys/values, the part with the most room for an
    # indexing error (5 sites, 6 tables, orders 2 and 3, conv taps).
    mask = make_causal_mask(n)
    ek = model.apply(
        {"params": lm}, tokens, mask, False, method=SimpleAttentionNetwork._engram_kv
    )
    if ek is not None:
        k, v = ek
        rec["stages"]["engram_k"] = blob.add(k)
        rec["stages"]["engram_v"] = blob.add(v)
        print(f"  engram k {tuple(np.asarray(k).shape)} v {tuple(np.asarray(v).shape)}")
    orders, eheads, sub_dim = engram_geometry(cfg)
    rec["geometry"]["engram_orders"] = list(orders)
    rec["geometry"]["engram_heads"] = int(eheads)
    rec["geometry"]["engram_sub_dim"] = int(sub_dim)

    # Stage 4 — logits. The tied head is the embedding sliced to out_vocab.
    t2 = time.time()
    logits = model.apply({"params": lm}, tokens)
    logits = np.asarray(logits, dtype=np.float32)
    rec["stages"]["logits"] = blob.add(logits[0])
    print(f"  logits {logits.shape}  {time.time() - t2:.1f}s")

    # A compact, high-signal summary: the greedy argmax at each position, so a
    # Rust divergence shows up as a position index rather than a float diff.
    rec["argmax"] = [int(i) for i in np.argmax(logits[0], axis=-1)]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_F32.write_bytes(bytes(blob.buf))
    OUT_JSON.write_text(json.dumps(rec, indent=1) + "\n")
    print(
        f"wrote {OUT_JSON} and {OUT_F32} "
        f"({len(blob.buf) / 1024 / 1024:.1f} MB of f32)"
    )


if __name__ == "__main__":
    main()
