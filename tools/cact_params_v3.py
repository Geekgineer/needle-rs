#!/usr/bin/env python3
"""Rebuild a Needle 3 Flax parameter tree straight from `needle3.cact`.

This is the canon check. Per-tensor parity cannot catch a canon error: the
container's directory is nameless and positional, so a *correct* tensor
landing in the wrong slot still passes every individual comparison. The only
way to pin the ordering is to invert `export._tensors`, hand the result back
to upstream's own model, and see whether it produces coherent output.

Everything here mirrors `_tensors` in `needle/model/export.py` exactly, in
the same order. Reference is pinned at `dd85774` on branch `needle3-oracle`.

    PYTHONPATH=needle:tools JAX_PLATFORMS=cpu .venv-parity/bin/python \
        tools/cact_params_v3.py
"""

import numpy as np

from needle.model import export
from needle.model.architecture import TransformerConfig, head_dims

CONTAINER = "weights/needle3.cact"


def config_from_meta(meta):
    """A TransformerConfig matching what the container declares."""
    cfg = TransformerConfig(
        vocab_size=meta["vocab_size"],
        d_model=meta["d_model"],
        num_heads=meta["num_heads"],
        num_kv_heads=meta["num_kv_heads"],
        num_layers=meta["num_layers"],
        max_seq_len=meta["max_seq_len"],
    )
    # Anything the container states explicitly wins over the config default,
    # so a future checkpoint with different extras still rebuilds correctly.
    for attr, key in (
        ("sliding_window", "sliding_window"),
        ("qkv_conv_taps", "qkv_conv_taps"),
        ("mhc_lanes", "mhc_lanes"),
        ("rope_theta", "rope_theta"),
        ("engram_slots", "engram_slots"),
    ):
        if key in meta and hasattr(cfg, attr):
            setattr(cfg, attr, meta[key])
    if "engram_layers" in meta:
        cfg.engram_layers = tuple(meta["engram_layers"])
    if "engram_orders" in meta:
        cfg.engram_orders = tuple(meta["engram_orders"])
    if "global_layers" in meta:
        cfg.global_layers = tuple(meta["global_layers"])
    return cfg


def rebuild(container=CONTAINER):
    """Return `(params, config, meta)` reconstructed from the container."""
    meta, tensors = export.read_export(container)
    cfg = config_from_meta(meta)
    qk_hd, v_hd, orders, eheads, sub_dim, sites = export._geometry(cfg)
    num_tables = len(orders) * eheads
    L = cfg.num_layers
    taps_n = int(getattr(cfg, "qkv_conv_taps", 0))

    it = iter(range(len(tensors)))
    nxt = lambda: tensors[next(it)]  # noqa: E731 — walking the canon in order

    params = {}
    params["embedding"] = {"embedding": nxt()}

    # Per-layer tensors are stacked along a leading layer axis in the Flax
    # tree, and `_tensors` exports slice `i` of each. Collect, then stack.
    acc = {
        k: []
        for k in (
            "norm_in", "q_proj", "k_proj", "v_proj", "q_taps", "k_taps", "v_taps",
            "q_norm", "k_norm", "gate_proj", "out_proj", "post_norm", "attn_gate",
            "pre_hada", "d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b",
            "w3a", "w3b", "cond_v", "cond_u",
        )
    }
    for _ in range(L):
        acc["norm_in"].append(nxt())
        acc["q_proj"].append(nxt())
        acc["k_proj"].append(nxt())
        acc["v_proj"].append(nxt())
        if taps_n:
            acc["q_taps"].append(nxt())
            acc["k_taps"].append(nxt())
            acc["v_taps"].append(nxt())
        acc["q_norm"].append(nxt())
        acc["k_norm"].append(nxt())
        acc["gate_proj"].append(nxt())
        acc["out_proj"].append(nxt())
        acc["post_norm"].append(nxt())
        acc["attn_gate"].append(nxt())
        acc["pre_hada"].append(nxt())
        for k in ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b",
                  "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u"):
            acc[k].append(nxt())

    st = lambda k: np.stack(acc[k])                    # noqa: E731
    stT = lambda k: np.stack([t.T for t in acc[k]])    # noqa: E731 — kernels were transposed on export

    self_attn = {
        "q_proj": {"kernel": stT("q_proj")},
        "k_proj": {"kernel": stT("k_proj")},
        "v_proj": {"kernel": stT("v_proj")},
        "gate_proj": {"kernel": stT("gate_proj")},
        "out_proj": {"kernel": stT("out_proj")},
        "q_norm": {"scale": st("q_norm")},
        "k_norm": {"scale": st("k_norm")},
    }
    if taps_n:
        self_attn["q_taps"] = st("q_taps")
        self_attn["k_taps"] = st("k_taps")
        self_attn["v_taps"] = st("v_taps")

    block = {
        "ZCRMSNorm_0": {"scale": st("norm_in")},
        "self_attn": self_attn,
        "post_attn_norm": {"scale": st("post_norm")},
        # exported as shape (1,) per layer; the tree holds a scalar per layer
        "attn_gate": np.stack([np.asarray(t).reshape(()) for t in acc["attn_gate"]]),
        "pre_hada_norm": {"scale": st("pre_hada")},
        "hadamard_mlp": {
            k: st(k)
            for k in ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b",
                      "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u")
        },
    }

    stack = {"layers": {"block": block}}
    for name in ("mhc_a_pre", "mhc_a_post", "mhc_a_res",
                 "mhc_b_pre", "mhc_b_post", "mhc_b_res"):
        stack[name] = nxt()
    # Export does `phi.transpose(0, 2, 1).reshape(L * k, nC)`, where k is the
    # trailing axis. It is `mhc_lanes` for pre/post but `mhc_lanes ** 2` for
    # res, which mixes lane against lane — so derive k rather than assume it.
    for name in ("mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"):
        flat = nxt()
        rows, nC = flat.shape
        assert rows % L == 0, f"{name}: {rows} rows is not a multiple of {L} layers"
        k = rows // L
        stack[name] = flat.reshape(L, k, nC).transpose(0, 2, 1)

    # hada_p1 / hada_p2 are derived permutations, not learned parameters. The
    # model recomputes them from d_model, so they are consumed and dropped.
    _hada_p1, _hada_p2 = nxt(), nxt()

    for s in range(len(sites)):
        tables = nxt().reshape(num_tables, cfg.engram_slots, sub_dim)
        key_k = nxt().T
        val_k = nxt().T
        taps = nxt()
        params[f"engrams_{s}"] = {
            "embedding": tables,
            "key_proj": {"kernel": key_k},
            "value_proj": {"kernel": val_k},
            "taps": taps,
        }

    stack["final_norm"] = {"scale": nxt()}
    params["stack"] = stack

    # Heads: a manifest of codes, then each head's tensors in canon order.
    from needle.model.architecture import HEADS

    manifest = np.asarray(nxt()).astype(int).tolist()
    by_code = {h.code: h for h in HEADS}
    params["_heads"] = {by_code[c].key: c for c in manifest if c in by_code}

    # Rebuild each head's parameter tree, inverting ProbeHead.export:
    # probes, gain, query, row_bias, proj, bias — plus calibration for the
    # router. `cells` is one per layer plus the input embedding.
    l1 = cfg.num_layers + 1
    for code in manifest:
        head = by_code.get(code)
        if head is None:
            continue
        probes = nxt()          # (l1 * k, d)
        gain = nxt()            # (l1, k)
        query = nxt()           # (q, d)
        row_bias = nxt()        # (q, l1, k)
        proj = nxt()            # (out, q * d) — exported transposed
        bias = nxt()            # (out,)
        k = probes.shape[0] // l1
        d = probes.shape[1]
        q = query.shape[0]
        tree = {
            "probes": probes.reshape(l1, k, d),
            "gain": np.asarray(gain).reshape(l1, k),
            "query": query,
            "row_bias": np.asarray(row_bias).reshape(q, l1, k),
            "proj": {"kernel": proj.T, "bias": np.asarray(bias)},
        }
        if head.key == "router_head":
            tree["calibration"] = np.asarray(nxt())
        params[head.key] = tree

    params["_head_tensors"] = [tensors[i] for i in it]

    return params, cfg, meta


def main():
    params, cfg, meta = rebuild()
    print(f"rebuilt from {CONTAINER}")
    print(f"  layers        {cfg.num_layers} x {cfg.d_model}")
    print(f"  heads         {cfg.num_heads} q / {cfg.num_kv_heads} kv, "
          f"qk {head_dims(cfg)[0]} / v {head_dims(cfg)[1]}")
    emb = params["embedding"]["embedding"]
    print(f"  embedding     {emb.shape}")
    q = params["stack"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"]
    print(f"  q_proj kernel {q.shape}  (L, d_model, heads*qk)")
    phi = params["stack"]["mhc_phi_pre"]
    print(f"  mhc_phi_pre   {phi.shape}  (L, nC, lanes)")
    eg = params["engrams_0"]["embedding"]
    print(f"  engram tables {eg.shape}  (tables, slots, sub_dim)")
    print(f"  head tensors  {len(params['_head_tensors'])} remaining")


if __name__ == "__main__":
    main()
