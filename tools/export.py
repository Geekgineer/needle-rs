#!/usr/bin/env python3
"""
Export a Needle pickle checkpoint to SafeTensors + vocabulary text file.

Usage (from needle-rust/ root):
    PYTHONPATH=needle python3 tools/export.py \
        --checkpoint needle/checkpoints/needle.pkl \
        --output-dir weights/

Outputs:
    weights/needle.safetensors   -- quantized weights in SafeTensors format
    weights/vocab.txt            -- one SentencePiece piece per line

Tensor naming convention (matches needle-infer/src/engine.rs):
    embedding                              [vocab, d_model] BF16
    encoder_final_norm                     [d_model] F32
    decoder_final_norm                     [d_model] F32
    encoder.{i}.norm                       [d_model] F32
    encoder.{i}.self_attn_gate             [] scalar F32
    encoder.{i}.self_attn.wq              INT4 + .scale F32
    encoder.{i}.self_attn.wk/wv/wo        INT4 + .scale F32
    encoder.{i}.self_attn.q_norm/k_norm   [head_dim] F32
    decoder.{i}.self_attn_norm            [d_model] F32
    decoder.{i}.cross_attn_norm           [d_model] F32
    decoder.{i}.self_attn_gate            [] scalar F32
    decoder.{i}.cross_attn_gate           [] scalar F32
    decoder.{i}.self_attn.wq/wk/wv/wo    INT4 + .scale F32
    decoder.{i}.self_attn.q_norm/k_norm  [head_dim] F32
    decoder.{i}.cross_attn.wq/wk/wv/wo  INT4 + .scale F32
    decoder.{i}.cross_attn.q_norm/k_norm [head_dim] F32

INT4 packing (row-major, matching quant.rs):
    data[pair * out_feat + o]  where pair=row//2
    byte = lo_nibble(row 2*pair) | hi_nibble(row 2*pair+1)
    scale[g * out_feat + o] = max(|group|) / 7.0 >= 1e-8
"""

import argparse
import json
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "needle"))

GROUP_SIZE = 32
SCALE_MIN = 1e-8


# ── Quantization ──────────────────────────────────────────────────────────────

def quantize_int4(w: np.ndarray) -> tuple[bytes, np.ndarray]:
    """
    Symmetric group-wise INT4 quantization with ROW-MAJOR packing.
    Matches quant.rs QuantizedWeight::quantize exactly.

    w: [in_feat, out_feat] float32
    Returns: (packed_bytes, scales [num_groups, out_feat] float32)
    """
    assert w.ndim == 2, f"expected 2D, got {w.shape}"
    in_feat, out_feat = w.shape
    gs = min(GROUP_SIZE, in_feat)
    pad = (gs - in_feat % gs) % gs
    if pad:
        w = np.pad(w, ((0, pad), (0, 0)))
    in_padded, num_groups = w.shape[0], w.shape[0] // gs

    w_g = w.reshape(num_groups, gs, out_feat)           # [G, gs, O]
    scales = np.maximum(np.abs(w_g).max(axis=1) / 7.0, SCALE_MIN)  # [G, O]
    w_q = np.clip(np.round(w_g / scales[:, None, :]), -8, 7).astype(np.int8)
    w_q = w_q.reshape(in_padded, out_feat)              # [in_padded, O]

    # Row-major packing: for each pair of input rows, pack into one byte per output.
    num_pairs = in_padded // 2
    w_pairs = w_q.reshape(num_pairs, 2, out_feat)       # [pairs, 2, O]
    lo = w_pairs[:, 0, :].astype(np.uint8) & 0x0F      # [pairs, O]
    hi = (w_pairs[:, 1, :].astype(np.uint8) & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)                 # [pairs, O] C-order

    return packed.tobytes(), scales.astype(np.float32)


def to_bf16(arr: np.ndarray) -> bytes:
    """Convert float32 → BF16 (truncate lower 16 bits of the IEEE mantissa)."""
    return arr.astype(np.float32).view(np.uint32).astype(np.uint32).__rshift__(16).astype(np.uint16).tobytes()


# ── Param extraction ──────────────────────────────────────────────────────────

def extract_tensors(params: dict, num_enc: int, num_dec: int):
    """Yield (name, numpy_array) for every tensor the Rust engine needs."""
    # Shortcut references into the scan-stacked param dicts
    enc_p = params["encoder"]["layers"]["EncoderBlock_0"]
    dec_p = params["decoder"]["layers"]["DecoderBlock_0"]

    # ── Top-level ──
    yield "embedding", np.array(params["embedding"]["embedding"], np.float32)
    yield "encoder_final_norm", np.array(params["encoder"]["final_norm"]["scale"], np.float32)
    yield "decoder_final_norm", np.array(params["decoder"]["ZCRMSNorm_0"]["scale"], np.float32)

    # ── Encoder layers ──
    for i in range(num_enc):
        pf = f"encoder.{i}"
        yield f"{pf}.norm",            np.array(enc_p["ZCRMSNorm_0"]["scale"][i], np.float32)
        yield f"{pf}.self_attn_gate",  np.array(enc_p["attn_gate"][i], np.float32).reshape(1)
        sa = enc_p["self_attn"]
        for proj, key in [("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv"), ("out_proj", "wo")]:
            yield f"{pf}.self_attn.{key}", np.array(sa[proj]["kernel"][i], np.float32)
        yield f"{pf}.self_attn.q_norm", np.array(sa["q_norm"]["scale"][i], np.float32)
        yield f"{pf}.self_attn.k_norm", np.array(sa["k_norm"]["scale"][i], np.float32)

    # ── Decoder layers ──
    for i in range(num_dec):
        pf = f"decoder.{i}"
        yield f"{pf}.self_attn_norm",  np.array(dec_p["ZCRMSNorm_0"]["scale"][i], np.float32)
        yield f"{pf}.cross_attn_norm", np.array(dec_p["ZCRMSNorm_1"]["scale"][i], np.float32)
        yield f"{pf}.self_attn_gate",  np.array(dec_p["self_attn_gate"][i], np.float32).reshape(1)
        yield f"{pf}.cross_attn_gate", np.array(dec_p["cross_attn_gate"][i], np.float32).reshape(1)
        for attn_name in ("self_attn", "cross_attn"):
            at = dec_p[attn_name]
            for proj, key in [("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv"), ("out_proj", "wo")]:
                yield f"{pf}.{attn_name}.{key}", np.array(at[proj]["kernel"][i], np.float32)
            yield f"{pf}.{attn_name}.q_norm", np.array(at["q_norm"]["scale"][i], np.float32)
            yield f"{pf}.{attn_name}.k_norm", np.array(at["k_norm"]["scale"][i], np.float32)


QUANT_SUFFIXES = (".wq", ".wk", ".wv", ".wo")


def should_quantize(name: str, arr: np.ndarray) -> bool:
    return any(name.endswith(s) for s in QUANT_SUFFIXES) and arr.ndim == 2


# ── SafeTensors writer ────────────────────────────────────────────────────────

def write_safetensors(tensors: dict, path: str):
    """tensors: { name: (dtype_str, shape_list, raw_bytes) }"""
    data_parts = []
    offset = 0
    header = {}
    for name, (dtype, shape, data) in tensors.items():
        end = offset + len(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        data_parts.append(data)
        offset = end

    header_json = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for part in data_parts:
            f.write(part)

    mb = (8 + len(header_json) + offset) / 1024 / 1024
    print(f"  Wrote {path}  ({mb:.1f} MB, {len(tensors)} tensors)")


# ── Main export ───────────────────────────────────────────────────────────────

def export_checkpoint(ckpt_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {ckpt_path} ...")
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    config = data.get("config", {})
    num_enc = config.get("num_encoder_layers", 12)
    num_dec = config.get("num_decoder_layers", 8)
    print(f"  Model: {num_enc} enc + {num_dec} dec layers, "
          f"d={config.get('d_model', 512)}, vocab={config.get('vocab_size', 8192)}")

    params = data["params"]

    tensors_out: dict[str, tuple] = {}
    n_quant = 0

    for name, arr in extract_tensors(params, num_enc, num_dec):
        if should_quantize(name, arr):
            in_f, out_f = arr.shape
            gs = min(GROUP_SIZE, in_f)
            in_pad = ((in_f + gs - 1) // gs) * gs
            num_g = in_pad // gs
            packed, scales = quantize_int4(arr)
            # Store shape as [num_pairs, out_feat] for informational purposes
            tensors_out[name] = ("I4", [in_pad // 2, out_f], packed)
            tensors_out[f"{name}.scale"] = ("F32", [num_g, out_f], scales.tobytes())
            n_quant += 1
        elif name == "embedding":
            # Embedding stored as BF16 to halve size (engine converts back to f32 on load)
            tensors_out[name] = ("BF16", list(arr.shape), to_bf16(arr))
        else:
            # Norms, gates, q_norm, k_norm → keep as F32
            tensors_out[name] = ("F32", list(arr.shape), arr.tobytes())

    st_path = os.path.join(output_dir, "needle.safetensors")
    print(f"\nExporting {len(tensors_out)} tensors ({n_quant} quantized projections) ...")
    write_safetensors(tensors_out, st_path)

    # ── Vocabulary ──
    vocab_path = os.path.join(output_dir, "vocab.txt")
    try:
        from needle.dataset.tokenizer import NeedleTokenizer
        tok = NeedleTokenizer()
        pieces = [tok.sp.id_to_piece(i) for i in range(tok.sp.get_piece_size())]
        with open(vocab_path, "w", encoding="utf-8") as f:
            for p in pieces:
                f.write(p + "\n")
        print(f"  Wrote {vocab_path}  ({len(pieces)} pieces)")
    except Exception as e:
        print(f"  Warning: vocab export failed: {e}")

    print(f"\nDone. Load in Rust:")
    print(f'  NeedleEngine::load("{st_path}", "{vocab_path}")')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", default="weights")
    args = ap.parse_args()
    export_checkpoint(args.checkpoint, args.output_dir)
