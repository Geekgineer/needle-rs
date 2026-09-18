#!/usr/bin/env python3
"""Build a real Needle 3 rung container, the way `needle build --layers N` does.

Calls the upstream pieces directly — load_checkpoint -> rung -> write_export —
rather than the CLI, so the 35 MB base archive is read from disk for its
tokenizer blob instead of being re-fetched.
"""
import os, sys
sys.path.insert(0, "needle")

from needle.model.run import load_checkpoint
from needle.model.finetune import rung
from needle.model.architecture import effective_kv_window
from needle.model.export import read_tokenizer_blob, write_export, read_layers
from needle.model.quantize import WEIGHT_BITS

CKPT = sys.argv[1]
BASE = sys.argv[2]
OUT_DIR = sys.argv[3]
DEPTHS = [int(x) for x in sys.argv[4].split(",")]

print(f"base archive : {BASE}  ({read_layers(BASE)} layers)")
tok = read_tokenizer_blob(BASE)
print(f"tokenizer    : {len(tok)} bytes")

params, config, _ = load_checkpoint(CKPT, return_run=True)
print(f"checkpoint   : {CKPT}  num_layers={config.num_layers} "
      f"global={tuple(config.global_layers)} engram={tuple(config.engram_layers)}")

os.makedirs(OUT_DIR, exist_ok=True)
for d in DEPTHS:
    p, c = rung(params, config, d)
    out = os.path.join(OUT_DIR, f"needle3-{d}l.cact")
    info = write_export(p, c, out, bits=WEIGHT_BITS,
                        tokenizer=tok, kv_window=effective_kv_window(c))
    print(f"  depth {d:>2}: {info['bytes']/1e6:7.2f} MB  {info['tensors']:>4} tensors  "
          f"global={tuple(c.global_layers)} engram={tuple(c.engram_layers)}  -> {out}")
