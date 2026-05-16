#!/usr/bin/env python3
"""
Generate test vectors for Needle Rust parity tests.

Creates a tiny deterministic model (d=16, 2 enc + 2 dec layers) using
float32 throughout so Rust can verify numerical parity.

Outputs JSON to stdout or --output path. The vectors cover:
  - ZCRMSNorm
  - RoPE
  - Quantization (INT4 round-trip)
  - All layer weights (for Rust to reconstruct the model)
  - Full forward pass result (pred_token, pred_token_q)
  - pred_token_q: argmax when Python also uses INT4-fake-quantized kernels
    so Rust INT4 forward must agree exactly (modulo FP accumulation order).

Usage:
    cd needle-rust/
    PYTHONPATH=needle python3 tools/gen_test_vectors.py --output tests/vectors.json

Prints param tree structure to stderr so you can verify Flax key paths.
"""

import argparse
import json
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + "/../needle")
from needle.model.architecture import (
    SimpleAttentionNetwork, TransformerConfig,
    precompute_rope_freqs, apply_rope, ZCRMSNorm,
)
from needle.model.quantize import _fake_quantize_int4


# ── tiny deterministic config ──────────────────────────────────────────────────
CFG = TransformerConfig(
    vocab_size=32,
    d_model=16,
    num_heads=4,
    num_kv_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=32,
    max_seq_len=16,
    rope_theta=10000.0,
    dtype="float32",
    no_feedforward=True,
)

SEED = 42
ENC_INPUT_IDS = jnp.array([[2, 5, 7, 3]], dtype=jnp.int32)   # [1, 4]
DEC_INPUT_IDS = jnp.array([[1]], dtype=jnp.int32)             # [1, 1]


def f32(x):
    return jnp.asarray(x, dtype=jnp.float32)


def to_list(x):
    return np.array(x, dtype=np.float64).tolist()


def get(params, *path):
    """Walk nested param dict by path components."""
    x = params
    for k in path:
        x = x[k]
    return x


def fake_quantize_params(params):
    """Replace every attention Dense kernel with its INT4 fake-quantized version.

    Only quantizes kernels inside q_proj / k_proj / v_proj / out_proj submodules
    (same targets as the Rust INT4 quantizer). Leaves embedding and norm scales
    untouched.
    """
    def walk(node, path=()):
        if isinstance(node, dict):
            return {k: walk(v, path + (k,)) for k, v in node.items()}
        # Quantize only Dense kernels inside attention projection submodules
        if (hasattr(node, 'shape') and
                path and path[-1] == "kernel" and
                len(path) >= 2 and
                path[-2] in ("q_proj", "k_proj", "v_proj", "out_proj")):
            if len(node.shape) == 2:
                # Standard 2-D kernel [in_feat, out_feat]
                return _fake_quantize_int4(node, group_size=32)
            if len(node.shape) == 3:
                # Scan-stacked kernel [num_layers, in_feat, out_feat]
                return jnp.stack(
                    [_fake_quantize_int4(node[i], group_size=32)
                     for i in range(node.shape[0])]
                )
        return node

    return walk(params)


def gen_vectors():
    rng = jax.random.PRNGKey(SEED)
    model = SimpleAttentionNetwork(CFG)

    # Init model
    rng, init_rng = jax.random.split(rng)
    dummy_src = jnp.ones((1, 4), dtype=jnp.int32)
    dummy_tgt = jnp.ones((1, 4), dtype=jnp.int32)
    variables = model.init({"params": init_rng}, dummy_src, dummy_tgt)
    params = variables["params"]

    # Print param tree to stderr
    print("\n=== Param tree (key → shape) ===", file=sys.stderr)
    leaves_with_path = jax.tree_util.tree_leaves_with_path(params)
    for path, leaf in leaves_with_path:
        path_str = "/".join(str(p.key) for p in path)
        print(f"  {path_str}: {leaf.shape}", file=sys.stderr)

    # ── ZCRMSNorm test vector ──────────────────────────────────────────────
    zc_x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    zc_scale = jnp.zeros(4, dtype=jnp.float32)
    eps = 1e-6
    zc_out = to_list((1.0 + zc_scale) * zc_x / jnp.sqrt(jnp.mean(zc_x ** 2) + eps))

    zc_scale_nonzero = jnp.array([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    zc_out_nonzero = to_list((1.0 + zc_scale_nonzero) * zc_x / jnp.sqrt(jnp.mean(zc_x ** 2) + eps))

    # ── RoPE test vector ──────────────────────────────────────────────────
    head_dim = CFG.d_model // CFG.num_heads  # 4
    cos_table, sin_table = precompute_rope_freqs(head_dim, 8, CFG.rope_theta)
    rope_q = jnp.arange(1, 5, dtype=jnp.float32).reshape(1, 1, 1, head_dim)
    rope_q_rotated = apply_rope(rope_q, cos_table, sin_table)

    # ── Quantization test vector ──────────────────────────────────────────
    rng, qrng = jax.random.split(rng)
    quant_w = jax.random.normal(qrng, (32, 16), dtype=jnp.float32)
    quant_w_dq = _fake_quantize_int4(quant_w, group_size=32)
    gs = min(32, quant_w.shape[0])
    quant_w_grouped = quant_w.reshape(-1, gs, quant_w.shape[1])
    quant_scales = jnp.max(jnp.abs(quant_w_grouped), axis=1) / 7.0
    quant_scales = jnp.maximum(quant_scales, 1e-8)
    quant_w_q = jnp.clip(jnp.round(quant_w_grouped / quant_scales[:, None, :]), -8, 7)
    quant_w_q = quant_w_q.reshape(-1, quant_w.shape[1])

    # ── Full forward pass (f32 weights) ───────────────────────────────────
    src = ENC_INPUT_IDS
    tgt = DEC_INPUT_IDS
    embed_scale = float(jnp.sqrt(jnp.float32(CFG.d_model)))

    # Encoder
    enc_out, _ = model.apply(variables, src, method=model.encode_text)   # [1, 4, 16]
    enc_out = f32(enc_out)

    # Embedding before encoder (from raw params)
    emb_weights = f32(params["embedding"]["embedding"])  # [vocab, d_model]
    enc_embed = emb_weights[src[0]] * embed_scale        # [4, 16]

    # Decoder — logits from model.decode()
    logits = f32(model.apply(variables, tgt, enc_out, method=model.decode))  # [1, 1, vocab]

    # Dec hidden state before lm_head
    def _decode_hidden(self, tgt, encoder_out):
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        return self.decoder(x, encoder_out, deterministic=True)

    dec_out = f32(model.apply(variables, tgt, enc_out, method=_decode_hidden))  # [1, 1, 16]
    dec_embed = emb_weights[tgt[0]] * embed_scale  # [1, 16]

    pred_token = int(jnp.argmax(logits[0, 0]))

    # ── INT4-fake-quantized forward pass (for exact Rust parity) ──────────
    # Apply the same INT4 quantize→dequantize to every attention kernel that
    # the Rust model does. The resulting pred_token_q / logits_q should match
    # the Rust INT4 forward pass up to floating-point accumulation order.
    q_params = fake_quantize_params(params)
    q_vars = {"params": q_params}

    enc_out_q, _ = model.apply(q_vars, src, method=model.encode_text)
    enc_out_q = f32(enc_out_q)
    logits_q = f32(model.apply(q_vars, tgt, enc_out_q, method=model.decode))
    pred_token_q = int(jnp.argmax(logits_q[0, 0]))

    print(f"\npred_token (f32):  {pred_token}", file=sys.stderr)
    print(f"pred_token_q (INT4 fakeq): {pred_token_q}", file=sys.stderr)

    # ── Multi-step decode with INT4-fakeq weights (for KV-cache parity) ──
    # Python re-encodes the full prefix each step (no KV cache).
    # Rust uses incremental KV-cache decode. Results must agree, which validates
    # the Rust KV cache implementation against Python's full attention.
    ms_tokens = [int(DEC_INPUT_IDS[0, 0])]  # start from dec_start_id
    ms_steps = []
    for _step_i in range(3):
        tgt_so_far = jnp.array([ms_tokens], dtype=jnp.int32)
        # Returns [1, len(ms_tokens), vocab]; take last position
        sl = f32(model.apply(q_vars, tgt_so_far, enc_out_q, method=model.decode))
        sl_last = sl[0, -1]
        top5_idx = jnp.argsort(-sl_last)[:5].tolist()
        top5_log = [float(sl_last[int(i)]) for i in top5_idx]
        margin = top5_log[0] - top5_log[1]
        ms_steps.append({
            "input_token": ms_tokens[-1],
            "top5_tokens": [int(t) for t in top5_idx],
            "top5_logits": top5_log,
            "pred_token": int(top5_idx[0]),
            "margin": float(margin),
        })
        ms_tokens.append(int(top5_idx[0]))
        print(f"  step {_step_i}: input={ms_tokens[-2]} pred={ms_tokens[-1]} margin={margin:.4f}", file=sys.stderr)

    # ── Extract model weights for Rust reconstruction ─────────────────────
    enc_p = params["encoder"]["layers"]["EncoderBlock_0"]
    dec_p = params["decoder"]["layers"]["DecoderBlock_0"]

    def enc_layer_weights(i):
        return {
            "norm": to_list(f32(enc_p["ZCRMSNorm_0"]["scale"][i])),
            "self_attn_gate": float(f32(enc_p["attn_gate"][i])),
            "self_attn": {
                "q_proj": to_list(f32(enc_p["self_attn"]["q_proj"]["kernel"][i])),   # [d, q_dim]
                "k_proj": to_list(f32(enc_p["self_attn"]["k_proj"]["kernel"][i])),   # [d, kv_dim]
                "v_proj": to_list(f32(enc_p["self_attn"]["v_proj"]["kernel"][i])),
                "out_proj": to_list(f32(enc_p["self_attn"]["out_proj"]["kernel"][i])),
                "q_norm": to_list(f32(enc_p["self_attn"]["q_norm"]["scale"][i])),
                "k_norm": to_list(f32(enc_p["self_attn"]["k_norm"]["scale"][i])),
            },
        }

    def dec_layer_weights(i):
        return {
            "self_attn_norm": to_list(f32(dec_p["ZCRMSNorm_0"]["scale"][i])),
            "cross_attn_norm": to_list(f32(dec_p["ZCRMSNorm_1"]["scale"][i])),
            "self_attn_gate": float(f32(dec_p["self_attn_gate"][i])),
            "cross_attn_gate": float(f32(dec_p["cross_attn_gate"][i])),
            "self_attn": {
                "q_proj": to_list(f32(dec_p["self_attn"]["q_proj"]["kernel"][i])),
                "k_proj": to_list(f32(dec_p["self_attn"]["k_proj"]["kernel"][i])),
                "v_proj": to_list(f32(dec_p["self_attn"]["v_proj"]["kernel"][i])),
                "out_proj": to_list(f32(dec_p["self_attn"]["out_proj"]["kernel"][i])),
                "q_norm": to_list(f32(dec_p["self_attn"]["q_norm"]["scale"][i])),
                "k_norm": to_list(f32(dec_p["self_attn"]["k_norm"]["scale"][i])),
            },
            "cross_attn": {
                "q_proj": to_list(f32(dec_p["cross_attn"]["q_proj"]["kernel"][i])),
                "k_proj": to_list(f32(dec_p["cross_attn"]["k_proj"]["kernel"][i])),
                "v_proj": to_list(f32(dec_p["cross_attn"]["v_proj"]["kernel"][i])),
                "out_proj": to_list(f32(dec_p["cross_attn"]["out_proj"]["kernel"][i])),
                "q_norm": to_list(f32(dec_p["cross_attn"]["q_norm"]["scale"][i])),
                "k_norm": to_list(f32(dec_p["cross_attn"]["k_norm"]["scale"][i])),
            },
        }

    vectors = {
        "config": {
            "vocab_size": CFG.vocab_size,
            "d_model": CFG.d_model,
            "num_heads": CFG.num_heads,
            "num_kv_heads": CFG.num_kv_heads,
            "num_encoder_layers": CFG.num_encoder_layers,
            "num_decoder_layers": CFG.num_decoder_layers,
            "rope_theta": CFG.rope_theta,
            "embed_scale": embed_scale,
        },
        "inputs": {
            "enc_ids": ENC_INPUT_IDS[0].tolist(),
            "dec_start_id": int(DEC_INPUT_IDS[0, 0]),
        },
        "zc_rms_norm": {
            "x": zc_x.tolist(),
            "scale_zero": zc_scale.tolist(),
            "out_zero": zc_out,
            "scale_nonzero": zc_scale_nonzero.tolist(),
            "out_nonzero": zc_out_nonzero,
            "eps": eps,
        },
        "rope": {
            "head_dim": head_dim,
            "cos_table_pos0": to_list(cos_table[0]),
            "sin_table_pos0": to_list(sin_table[0]),
            "cos_table_pos1": to_list(cos_table[1]),
            "sin_table_pos1": to_list(sin_table[1]),
            "q_before": to_list(rope_q[0, 0, 0]),
            "q_after_pos0": to_list(rope_q_rotated[0, 0, 0]),
        },
        "quantization": {
            "w_shape": list(quant_w.shape),
            "w": to_list(quant_w),
            "scales": to_list(quant_scales),
            "w_quantized_int": to_list(quant_w_q),
        },
        "weights": {
            "embedding": to_list(f32(params["embedding"]["embedding"])),  # [vocab, d_model]
            "encoder_final_norm": to_list(f32(params["encoder"]["final_norm"]["scale"])),  # [d_model]
            "decoder_final_norm": to_list(f32(params["decoder"]["ZCRMSNorm_0"]["scale"])),  # [d_model]
            "encoder_layers": [enc_layer_weights(i) for i in range(CFG.num_encoder_layers)],
            "decoder_layers": [dec_layer_weights(i) for i in range(CFG.num_decoder_layers)],
        },
        "forward": {
            "enc_embed": to_list(enc_embed),          # [4, 16]
            "enc_out": to_list(enc_out[0]),            # [4, 16]
            "enc_out_q": to_list(enc_out_q[0]),        # [4, 16] encoder hidden with INT4-fakeq weights
            "dec_embed": to_list(dec_embed),           # [1, 16]
            "dec_out": to_list(dec_out[0]),            # [1, 16]
            "logits": to_list(logits[0, 0]),           # [vocab] — Python f32 weights
            "pred_token": pred_token,                  # argmax of f32-weight logits
            "logits_q": to_list(logits_q[0, 0]),       # [vocab] — Python INT4-fakeq weights
            "pred_token_q": pred_token_q,              # argmax of INT4-fakeq logits (Rust must match)
        },
        "multi_step": {
            "dec_tokens": ms_tokens,
            "steps": ms_steps,
        },
    }
    return vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    vecs = gen_vectors()
    out_str = json.dumps(vecs, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(out_str)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(out_str)
