#!/usr/bin/env python3
"""
Generate real-model parity vectors from the actual Needle checkpoint.

Loads needle.pkl, applies INT4 fake-quantization (same as Rust), runs encode +
5 greedy decode steps using fixed token IDs, and saves per-step reference data.

Usage (from needle-rust/ root):
    PYTHONPATH=needle python3 tools/gen_real_vectors.py \
        --checkpoint needle/checkpoints/needle.pkl \
        --output tests/real_vectors.json

Outputs:
    tests/real_vectors.json — reference data for real_parity.rs
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "needle"))
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
from needle.model.quantize import _fake_quantize_int4


def f32(x):
    return jnp.asarray(x, dtype=jnp.float32)


def to_list(x):
    return np.array(x, dtype=np.float64).tolist()


def fake_quantize_params(params):
    """INT4 fake-quantize every attention projection kernel (same as Rust engine)."""
    def walk(node, path=()):
        if isinstance(node, dict):
            return {k: walk(v, path + (k,)) for k, v in node.items()}
        if (hasattr(node, 'shape') and
                path and path[-1] == "kernel" and
                len(path) >= 2 and
                path[-2] in ("q_proj", "k_proj", "v_proj", "out_proj")):
            if len(node.shape) == 2:
                return _fake_quantize_int4(node, group_size=32)
            if len(node.shape) == 3:
                return jnp.stack([_fake_quantize_int4(node[i], group_size=32)
                                  for i in range(node.shape[0])])
        return node
    return walk(params)


def gen_real_vectors(ckpt_path: str, num_steps: int = 5) -> dict:
    print(f"Loading {ckpt_path} ...", file=sys.stderr)
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    config_dict = data.get("config", {})
    cfg = TransformerConfig(
        vocab_size=config_dict.get("vocab_size", 8192),
        d_model=config_dict.get("d_model", 512),
        num_heads=config_dict.get("num_heads", 8),
        num_kv_heads=config_dict.get("num_kv_heads", 4),
        num_encoder_layers=config_dict.get("num_encoder_layers", 12),
        num_decoder_layers=config_dict.get("num_decoder_layers", 8),
        d_ff=config_dict.get("d_ff", 2048),
        max_seq_len=config_dict.get("max_seq_len", 1024),
        rope_theta=config_dict.get("rope_theta", 10000.0),
        dtype="float32",
        no_feedforward=config_dict.get("no_feedforward", True),
    )
    print(f"  Config: d={cfg.d_model}, {cfg.num_encoder_layers}enc+{cfg.num_decoder_layers}dec, "
          f"vocab={cfg.vocab_size}", file=sys.stderr)

    model = SimpleAttentionNetwork(cfg)
    params = data["params"]

    # Build Flax variables
    variables = {"params": params}

    # Fixed encoder token IDs (BOS=2, TOOLS=5, then arbitrary token IDs)
    # These same IDs are used in real_parity.rs — no tokenization required.
    ENC_IDS = [2, 5, 100, 200, 300, 1]  # BOS TOOLS tok tok tok EOS
    DEC_START = 1  # EOS token starts decoder

    src = jnp.array([ENC_IDS], dtype=jnp.int32)

    print("Applying INT4 fake-quantization to attention kernels ...", file=sys.stderr)
    q_params = fake_quantize_params(params)
    q_vars = {"params": q_params}

    # Encode
    enc_out_q, _ = model.apply(q_vars, src, method=model.encode_text)
    enc_out_q = f32(enc_out_q)
    print(f"  Encoder output shape: {enc_out_q.shape}", file=sys.stderr)

    # Multi-step greedy decode
    tokens = [DEC_START]
    steps = []
    for step_i in range(num_steps):
        tgt_so_far = jnp.array([tokens], dtype=jnp.int32)
        sl = f32(model.apply(q_vars, tgt_so_far, enc_out_q, method=model.decode))
        sl_last = sl[0, -1]  # [vocab]
        top5_idx = jnp.argsort(-sl_last)[:5].tolist()
        top5_log = [float(sl_last[int(i)]) for i in top5_idx]
        margin = top5_log[0] - top5_log[1]
        pred = int(top5_idx[0])
        steps.append({
            "input_token": tokens[-1],
            "top5_tokens": [int(t) for t in top5_idx],
            "top5_logits": top5_log,
            "pred_token": pred,
            "margin": float(margin),
        })
        print(f"  step {step_i}: input={tokens[-1]} pred={pred} margin={margin:.4f}", file=sys.stderr)
        tokens.append(pred)

    vectors = {
        "enc_ids": ENC_IDS,
        "dec_start": DEC_START,
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "num_encoder_layers": cfg.num_encoder_layers,
            "num_decoder_layers": cfg.num_decoder_layers,
        },
        "enc_out_q": to_list(enc_out_q[0]),  # [enc_len, d_model]
        "dec_tokens": tokens,
        "steps": steps,
    }
    return vectors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="tests/real_vectors.json")
    ap.add_argument("--num-steps", type=int, default=5)
    args = ap.parse_args()

    vecs = gen_real_vectors(args.checkpoint, args.num_steps)
    out_str = json.dumps(vecs, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(out_str)
    print(f"Written to {args.output}", file=sys.stderr)
