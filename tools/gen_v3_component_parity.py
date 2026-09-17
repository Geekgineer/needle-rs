#!/usr/bin/env python3
"""Capture per-component oracles for the Needle 3 forward pass.

Each component is exercised through upstream's own Flax module with the real
weights from `needle3.cact`, on a deterministic input. That makes each Rust
kernel verifiable on its own instead of only through 20 layers of logits,
where a sign error in one place surfaces as "the tokens are wrong".

    JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
        tools/gen_v3_component_parity.py

Emits `tests/v3_component_vectors.json` + `.f32`.
"""

import json
import pathlib

import jax.numpy as jnp
import numpy as np

from cact_params_v3 import rebuild
from needle.model.architecture import (
    HadamardMLP,
    MultiHeadAttention,
    _hada_blocks,
    _hada_perms,
    head_dims,
    make_causal_mask,
    precompute_rope_freqs,
)

OUT_JSON = pathlib.Path("tests/v3_component_vectors.json")
OUT_F32 = pathlib.Path("tests/v3_component_vectors.f32")

SEQ = 4  # a handful of positions is enough; the MLP is position-wise


class Blob:
    def __init__(self):
        self.buf = bytearray()

    def add(self, arr):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        off = len(self.buf) // 4
        self.buf.extend(a.tobytes())
        return {"offset": off, "len": int(a.size), "shape": list(np.asarray(arr).shape)}


def main():
    params, cfg, meta = rebuild()
    blob = Blob()
    rec = {"geometry": {"d_model": cfg.d_model}, "dtype": "float32", "components": {}}

    n = 1 << (cfg.d_model - 1).bit_length()
    ba, bb = _hada_blocks(n)
    p1, p2 = _hada_perms(n, False)
    rec["geometry"].update(
        {"hada_n": int(n), "block_a": int(ba), "block_b": int(bb), "cond_rank": 8}
    )
    rec["components"]["hada_p1"] = blob.add(np.asarray(p1, np.float32))
    rec["components"]["hada_p2"] = blob.add(np.asarray(p2, np.float32))

    # --- HadamardMLP, layer 0 -------------------------------------------
    ha = params["stack"]["layers"]["block"]["hadamard_mlp"]
    layer0 = {k: jnp.asarray(v[0]) for k, v in ha.items()}

    rng = np.random.RandomState(20260917)
    x = rng.standard_normal((1, SEQ, cfg.d_model)).astype(np.float32) * 0.5
    # float32 deliberately. The reference defaults to bfloat16, which is what
    # upstream ships, but a bf16 oracle cannot distinguish "the Rust kernel is
    # wrong" from "bf16 has three decimal digits" — measured against the bf16
    # default this exact kernel reads as 9.7 relative error on near-zero
    # elements while being correct to 1.8e-6 against f32. Component parity
    # proves the arithmetic; end-to-end token parity proves the behaviour.
    mlp = HadamardMLP(d_model=cfg.d_model, dtype=jnp.float32)
    out = mlp.apply({"params": layer0}, jnp.asarray(x))

    rec["components"]["mlp_in"] = blob.add(x[0])
    rec["components"]["mlp_out"] = blob.add(np.asarray(out, np.float32)[0])
    for name in ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b",
                 "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u"):
        rec["components"][f"mlp_{name}"] = blob.add(np.asarray(layer0[name], np.float32))
    print(f"  hadamard_mlp  in {x.shape} -> out {tuple(np.asarray(out).shape)}")


    # --- MultiHeadAttention, layer 0 ------------------------------------
    # Exercised with a sliding-window mask, since 16 of the 20 layers are
    # local; the global layers differ only in which mask they are handed.
    qk_hd, v_hd = head_dims(cfg)
    sa = params["stack"]["layers"]["block"]["self_attn"]
    a0 = {}
    for k, v in sa.items():
        if isinstance(v, dict):
            a0[k] = {ik: jnp.asarray(iv[0]) for ik, iv in v.items()}
        else:
            a0[k] = jnp.asarray(v[0])

    T = 8
    ax = rng.standard_normal((1, T, cfg.d_model)).astype(np.float32) * 0.5
    mask = make_causal_mask(T)
    window = int(getattr(cfg, "sliding_window", 0))
    if window:
        pos = jnp.arange(T)
        band = ((pos[:, None] - pos[None, :]) < window)[None, None]
        local_mask = mask & band
    else:
        local_mask = mask
    cos, sin = precompute_rope_freqs(qk_hd, T, cfg.rope_theta)

    attn = MultiHeadAttention(
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        dtype=jnp.float32,
        qk_head_dim=qk_hd,
        v_head_dim=v_hd,
        qkv_conv_taps=int(getattr(cfg, "qkv_conv_taps", 0)),
    )
    aout = attn.apply({"params": a0}, jnp.asarray(ax), mask=local_mask, rope=(cos, sin))

    rec["geometry"].update(
        {
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "qk_head_dim": int(qk_hd),
            "v_head_dim": int(v_hd),
            "seq": T,
            "sliding_window": window,
            "qkv_conv_taps": int(getattr(cfg, "qkv_conv_taps", 0)),
            "rope_theta": float(cfg.rope_theta),
        }
    )
    rec["components"]["attn_in"] = blob.add(ax[0])
    rec["components"]["attn_out"] = blob.add(np.asarray(aout, np.float32)[0])
    rec["components"]["attn_rope_cos"] = blob.add(cos)
    rec["components"]["attn_rope_sin"] = blob.add(sin)
    for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"):
        rec["components"][f"attn_{name}"] = blob.add(
            np.asarray(a0[name]["kernel"], np.float32)
        )
    for name in ("q_norm", "k_norm"):
        rec["components"][f"attn_{name}"] = blob.add(
            np.asarray(a0[name]["scale"], np.float32)
        )
    for name in ("q_taps", "k_taps", "v_taps"):
        if name in a0:
            rec["components"][f"attn_{name}"] = blob.add(np.asarray(a0[name], np.float32))
    print(f"  attention     in {ax.shape} -> out {tuple(np.asarray(aout).shape)}")

    # Sanity: the learned factors are initialised to Walsh but trained, so
    # record how far they have drifted. If they were still exactly Walsh a
    # fast transform would apply; this says whether that shortcut is open.
    from needle.model.architecture import _walsh_matrix

    w = np.asarray(_walsh_matrix(ba), np.float32)
    drift = {
        k: float(np.abs(np.asarray(layer0[k], np.float32) - w).max())
        for k in ("w1a", "w1b", "w2a", "w2b", "w3a", "w3b")
    }
    rec["walsh_drift"] = drift
    print("  factor drift from Walsh:", {k: round(v, 4) for k, v in drift.items()})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_F32.write_bytes(bytes(blob.buf))
    OUT_JSON.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_F32} ({len(blob.buf) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
