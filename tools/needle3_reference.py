#!/usr/bin/env python3
"""A Needle 3 reference in numpy, read straight from `needle3.cact`.

A transcription of `needle/model/architecture.py` for one sequence, with no KV
cache and no incremental decode: every position is recomputed from the whole
prefix. It needs numpy and nothing else.

That is the point. `check_v3_canon.py` and the `gen_v3_*` generators run
upstream's own Flax modules, which need JAX, flax and a checkout of `needle`
pinned to the right commit. This script needs none of them, because a `.cact`
container already carries the geometry, the weights, the Hadamard permutations
and the tokenizer. A contributor who cannot install JAX can still answer the
question the canon check answers: does this container, read by this ordering,
produce the documented tool call?

    python3 tools/needle3_reference.py weights/needle3.cact \
        "What is the weather in Paris?"

    # exit 1 unless the answer names the tool, for use in a script
    python3 tools/needle3_reference.py weights/needle3.cact --expect get_weather

It is slow on purpose: clarity over speed, one position at a time against the
full prefix. Expect about six seconds per generated token at a 48-token prompt.
Being slow and independent is what makes it useful as a second opinion; the
JAX oracles stay the primary one, because they are upstream's own code.
"""
import struct
import sys

import numpy as np

HDR = "<48If"
REC = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(REC)
FP16, FP32, CQ, RAW = 1, 2, 3, 4
TAG = 0x05E12A84
ENGRAM_SEED = 0x9E3779B9
ENGRAM_PRIME = 0x01000193
HDR_FIELDS = ("tag", "num_tensors", "codebook_len", "kv_window", "kv_bits",
              "vocab", "out_vocab", "d_model", "num_heads", "num_kv_heads",
              "num_layers", "qk_head_dim", "v_head_dim", "max_seq_len",
              "hada_n", "mhc_lanes", "sliding_window", "gmask_lo", "gmask_hi",
              "qkv_conv_taps", "engram_slots", "engram_sub_dim",
              "num_engram_tables", "engram_conv_taps", "engram_conv_dilation",
              "engram_seed_heads", "num_orders")


def walsh(n):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def unpack_lsb(packed, bits, in_pad):
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << np.uint64(8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = (1 << bits) - 1
    for i in range(8):
        idx[..., i] = (word >> np.uint64(i * bits)) & np.uint64(mask)
    return idx.reshape(out, in_pad)


class Cact:
    def __init__(self, path):
        raw = open(path, "rb").read()
        self.raw = raw
        h = struct.unpack_from(HDR, raw, 0)
        assert h[0] == TAG, "not a needle 3 cact"
        self.h = dict(zip(HDR_FIELDS, h[:27]))
        self.orders = tuple(h[27:31][:h[26]])
        self.num_sites = h[31]
        self.sites = tuple(h[32:48][:self.num_sites])
        self.rope_theta = h[48]
        self.global_layers = tuple(
            i for i in range(self.h["num_layers"])
            if (h[17] | (h[18] << 32)) >> i & 1)
        off = struct.calcsize(HDR)
        cb_n = self.h["codebook_len"]
        self.codebook = np.frombuffer(raw[off:off + cb_n * 4], np.float32)
        off += cb_n * 4
        self.recs = []
        for _ in range(self.h["num_tensors"]):
            self.recs.append(struct.unpack_from(REC, raw, off))
            off += REC_SIZE
        self._cache = {}

    def cb(self, bits):
        starts = {2: 0, 3: 4, 4: 12}
        return self.codebook[starts[bits]:starts[bits] + (1 << bits)]

    def tensor(self, i):
        if i in self._cache:
            return self._cache[i]
        r = self.recs[i]
        dtype, ndim = r[0], r[1]
        shape = tuple(r[3:3 + ndim])
        offset, nbytes, group, bits = r[7], r[8], r[9], r[10]
        blob = self.raw[offset:offset + nbytes]
        if dtype == FP16:
            t = np.frombuffer(blob, np.float16).reshape(shape).astype(np.float32)
        elif dtype == FP32:
            t = np.frombuffer(blob, np.float32).reshape(shape).copy()
        elif dtype == RAW:
            t = blob
        else:
            out, in_dim = shape
            in_pad = (in_dim + group - 1) // group * group
            n_packed = out * in_pad * bits // 8
            packed = np.frombuffer(blob[:n_packed], np.uint8).reshape(out, -1)
            norms = np.frombuffer(blob[n_packed:], np.float16).reshape(
                out, in_pad // group).astype(np.float32)
            idx = unpack_lsb(packed, bits, in_pad)
            unit = self.cb(bits)[idx].reshape(out, in_pad // group, group)
            rot = unit * norms[:, :, None]
            t = (rot @ walsh(group)).reshape(out, in_pad)[:, :in_dim]
        self._cache[i] = t
        return t


def rms_unit(x, eps=1e-6):
    return x / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + eps)


def zcrms(x, scale, eps=1e-6):
    return (1 + scale) * x / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def silu(x):
    return x * sigmoid(x)


def sinkhorn(logits, iters=20):
    k = logits
    for _ in range(iters):
        k = k - np.log(np.sum(np.exp(k - k.max(-1, keepdims=True)), -1, keepdims=True)) - k.max(-1, keepdims=True)
        k = k - np.log(np.sum(np.exp(k - k.max(-2, keepdims=True)), -2, keepdims=True)) - k.max(-2, keepdims=True)
    return np.exp(k)


def shift_right(x, off):
    if off == 0:
        return x
    out = np.zeros_like(x)
    out[off:] = x[:x.shape[0] - off]
    return out


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    u = tokens.astype(np.uint32)
    stride = seed_heads or heads
    cols = []
    for oi, order in enumerate(orders):
        for hh in range(heads):
            seed = np.uint32((ENGRAM_SEED * (oi * stride + hh + 1)) & 0xFFFFFFFF)
            acc = np.full(u.shape, seed, np.uint32)
            for j in range(order):
                acc = (acc ^ shift_right(u, j)) * np.uint32(ENGRAM_PRIME)
            acc = acc ^ (acc >> np.uint32(15))
            cols.append((acc % np.uint32(slots)).astype(np.int64))
    return np.stack(cols, -1)


def kron_apply(z, a, b):
    lead = z.shape[:-1]
    z = z.reshape(*lead, a.shape[0], b.shape[0])
    z = np.einsum("...ij,ik,jl->...kl", z, a, b)
    return z.reshape(*lead, a.shape[0] * b.shape[0])


class Layout:
    """Positional canon of the v3 container."""

    def __init__(self, c):
        h = c.h
        self.c = c
        L = h["num_layers"]
        self.per_layer = 24 + (3 if h["qkv_conv_taps"] else 0)
        self.embedding = 0
        self.layer0 = 1
        self.mhc = 1 + L * self.per_layer
        self.hada_p = self.mhc + 9
        self.engram = self.hada_p + 2
        self.final_norm = self.engram + 4 * c.num_sites
        self.heads = self.final_norm + 1

    def layer(self, i, name):
        taps = self.c.h["qkv_conv_taps"]
        names = ["norm_in", "q_proj", "k_proj", "v_proj"]
        if taps:
            names += ["q_taps", "k_taps", "v_taps"]
        names += ["q_norm", "k_norm", "gate_proj", "out_proj", "post_norm",
                  "attn_gate", "pre_hada", "d1", "d2", "b2", "d3", "d4",
                  "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u"]
        return self.layer0 + i * self.per_layer + names.index(name)

    def mhc_t(self, name):
        names = ["a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res",
                 "phi_pre", "phi_post", "phi_res"]
        return self.mhc + names.index(name)

    def engram_t(self, site, name):
        names = ["tables", "key_proj", "value_proj", "taps"]
        return self.engram + site * 4 + names.index(name)


class Model:
    def __init__(self, path):
        self.c = Cact(path)
        self.lo = Layout(self.c)
        h = self.c.h
        self.d = h["d_model"]
        self.L = h["num_layers"]
        self.lanes = h["mhc_lanes"]
        self.heads = h["num_heads"]
        self.kv_heads = h["num_kv_heads"]
        self.qk = h["qk_head_dim"]
        self.vh = h["v_head_dim"]
        self.taps = h["qkv_conv_taps"]
        self.window = h["sliding_window"]
        self.hada_n = h["hada_n"]
        self.emb = self.c.tensor(self.lo.embedding)

    def t(self, i):
        return self.c.tensor(i)

    def rope(self, T):
        inv = 1.0 / (self.c.rope_theta ** (np.arange(0, self.qk, 2, dtype=np.float32) / self.qk))
        ang = np.outer(np.arange(T, dtype=np.float32), inv)
        return np.cos(ang), np.sin(ang)

    def engram_kv(self, tokens, mask):
        c, lo = self.c, self.lo
        T = len(tokens)
        orders = c.orders
        n_tables = c.h["num_engram_tables"]
        per_order = n_tables // len(orders)
        sub = c.h["engram_sub_dim"]
        idx = engram_indices(tokens, orders, per_order, c.h["engram_slots"],
                             c.h["engram_seed_heads"])
        ngram_ok = np.stack([mask_diag(mask, o - 1) for o in orders
                             for _ in range(per_order)], -1)
        dil = c.h["engram_conv_dilation"]
        n_taps = c.h["engram_conv_taps"]
        tap_ok = np.stack([mask_diag(mask, j * dil) for j in range(n_taps)])
        ks, vs = [], []
        for s in range(c.num_sites):
            tables = self.t(lo.engram_t(s, "tables")).reshape(n_tables, -1, sub)
            fetched = tables[np.arange(n_tables)[None, :], idx]  # (T, tables, sub)
            fetched = fetched * ngram_ok[..., None]
            e = fetched.reshape(T, n_tables * sub)
            k = e @ self.t(lo.engram_t(s, "key_proj")).T
            v = e @ self.t(lo.engram_t(s, "value_proj")).T
            taps = self.t(lo.engram_t(s, "taps"))
            v = sum(taps[j] * shift_right(v, j * dil) * tap_ok[j][:, None]
                    for j in range(n_taps))
            ks.append(k)
            vs.append(v)
        return np.stack(ks), np.stack(vs)

    def attention(self, x, i, mask, rope):
        lo = self.lo
        T = x.shape[0]
        q = x @ self.t(lo.layer(i, "q_proj")).T
        k = x @ self.t(lo.layer(i, "k_proj")).T
        v = x @ self.t(lo.layer(i, "v_proj")).T
        if self.taps:
            qt = self.t(lo.layer(i, "q_taps"))
            kt = self.t(lo.layer(i, "k_taps"))
            vt = self.t(lo.layer(i, "v_taps"))
            tap_ok = [None] + [mask_diag(mask, j)[:, None] for j in range(1, self.taps)]

            def tapped(z, j):
                s = shift_right(z, j)
                return s if tap_ok[j] is None else s * tap_ok[j]
            q = sum(qt[j] * tapped(q, j) for j in range(self.taps))
            k = sum(kt[j] * tapped(k, j) for j in range(self.taps))
            v = sum(vt[j] * tapped(v, j) for j in range(self.taps))
        q = q.reshape(T, self.heads, self.qk).transpose(1, 0, 2)
        k = k.reshape(T, self.kv_heads, self.qk).transpose(1, 0, 2)
        v = v.reshape(T, self.kv_heads, self.vh).transpose(1, 0, 2)
        q = zcrms(q, self.t(lo.layer(i, "q_norm")))
        k = zcrms(k, self.t(lo.layer(i, "k_norm")))
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        rep = self.heads // self.kv_heads
        k = np.repeat(k, rep, 0)
        v = np.repeat(v, rep, 0)
        logits = q @ k.transpose(0, 2, 1) / np.sqrt(np.float32(self.qk))
        logits = np.where(mask[None], logits, np.float32(-3.0e38))
        out = softmax(logits, -1) @ v
        out = out.transpose(1, 0, 2).reshape(T, self.heads * self.vh)
        out = out * sigmoid(x @ self.t(lo.layer(i, "gate_proj")).T)
        return out @ self.t(lo.layer(i, "out_proj")).T

    def hada(self, x, i):
        lo = self.lo
        n = self.hada_n
        g = lambda nm: self.t(lo.layer(i, nm))
        cond = 1 + softmax(x @ g("cond_v"), -1) @ g("cond_u")
        z = np.pad(x, ((0, 0), (0, n - self.d)))
        p1 = self.t(lo.hada_p).astype(np.int64)
        p2 = self.t(lo.hada_p + 1).astype(np.int64)
        z = kron_apply(g("d1") * z, g("w1a"), g("w1b"))[..., p1]
        z = kron_apply(silu(g("d2") * cond * z + g("b2")), g("w2a"), g("w2b"))[..., p2]
        z = kron_apply(g("d3") * z, g("w3a"), g("w3b"))
        return (g("d4") * z)[..., :self.d]

    def block(self, x, i, mask, rope, ekv, site):
        lo = self.lo
        if site is not None:
            ek, ev = ekv[0][site], ekv[1][site]
            alpha = sigmoid(np.sum(rms_unit(x) * rms_unit(ek), -1) / np.sqrt(np.float32(self.d)))
            x = x + alpha[:, None] * ev
        skip = x
        y = zcrms(x, self.t(lo.layer(i, "norm_in")))
        y = self.attention(y, i, mask, rope)
        y = zcrms(y, self.t(lo.layer(i, "post_norm")))
        x = skip + sigmoid(self.t(lo.layer(i, "attn_gate"))[0]) * y
        skip = x
        y = zcrms(x, self.t(lo.layer(i, "pre_hada")))
        return skip + self.hada(y, i)

    def forward(self, tokens, collect=False):
        c, lo = self.c, self.lo
        T = len(tokens)
        n = self.lanes
        mask = np.tril(np.ones((T, T), bool))
        pad_ok = (tokens != 0)
        mask = mask & pad_ok[None, :]
        band = (np.arange(T)[:, None] - np.arange(T)[None, :]) < self.window
        local = mask & band
        rope = self.rope(T)
        ekv = self.engram_kv(tokens, mask) if c.num_sites else None
        x0 = self.emb[tokens] * np.sqrt(np.float32(self.d))
        x = np.broadcast_to(x0[:, None, :], (T, n, self.d)).astype(np.float32).copy()
        lane = np.eye(n, dtype=np.float32)[np.arange(self.L) % n]
        a_pre, a_post, a_res = (self.t(lo.mhc_t(k)) for k in ("a_pre", "a_post", "a_res"))
        b_pre, b_post, b_res = (self.t(lo.mhc_t(k)) for k in ("b_pre", "b_post", "b_res"))
        phi_pre = self.t(lo.mhc_t("phi_pre")).reshape(self.L, n, -1).transpose(0, 2, 1)
        phi_post = self.t(lo.mhc_t("phi_post")).reshape(self.L, n, -1).transpose(0, 2, 1)
        phi_res = self.t(lo.mhc_t("phi_res")).reshape(self.L, n * n, -1).transpose(0, 2, 1)
        cells = [x0]
        for i in range(self.L):
            m = mask if (i in c.global_layers or not self.window) else local
            site = c.sites.index(i) if i in c.sites else None
            nx = rms_unit(x.reshape(T, n * self.d))
            pre_off = 8 * lane[i] - 4
            post_off = -4 * (1 - lane[i])
            hpre = sigmoid(a_pre[i] * (nx @ phi_pre[i]) + b_pre[i] + pre_off)
            u = np.einsum("tn,tnc->tc", hpre, x)
            y = self.block(u, i, m, rope, ekv, site) - u
            hpost = 2 * sigmoid(a_post[i] * (nx @ phi_post[i]) + b_post[i] + post_off)
            res = (nx @ phi_res[i]).reshape(T, n, n)
            hres = sinkhorn(a_res[i] * res + b_res[i])
            x = np.einsum("tij,tjc->tic", hres, x) + hpost[..., None] * y[:, None, :]
            if collect:
                cells.append(x.mean(1))
        h = zcrms(x.mean(1), self.t(lo.final_norm))
        logits = h @ self.emb[:c.h["out_vocab"] or c.h["vocab"]].T
        if collect:
            return logits, np.stack(cells, 1)  # (T, L+1, d)
        return logits

    def confidence(self, tokens):
        _, cells = self.forward(tokens, collect=True)
        lo = self.lo
        base = lo.heads + 1  # skip heads.manifest
        probes, gain, query, row_bias, proj, bias = (self.t(base + j) for j in range(6))
        d = self.d
        L1 = cells.shape[1]
        k = probes.shape[0] // L1
        probes = probes.reshape(L1, k, d)
        q = query.shape[0]
        keep = (tokens != 0).astype(np.float32)
        scores = np.einsum("tld,lkd->lkt", cells, probes) / np.sqrt(np.float32(d))
        scores = np.where(keep[None, None, :] > 0, scores, -np.inf)
        r = np.einsum("lkt,tld->lkd", softmax(scores, -1), cells)
        r = rms_unit(r) * gain[:, :, None]
        u = np.einsum("lkd,qd->qlk", r, query) / np.sqrt(np.float32(d)) + row_bias
        w = softmax(u.reshape(q, L1 * k), -1)
        pooled = np.einsum("qm,md->qd", w, r.reshape(L1 * k, d)).reshape(q * d)
        return float(pooled @ proj.T + bias)


def mask_diag(mask, offset):
    T = mask.shape[-1]
    if offset >= T:
        return np.zeros(T, np.float32)
    d = np.diagonal(mask, offset=-offset).astype(np.float32)
    if offset == 0:
        return d
    return np.concatenate([np.zeros(offset, np.float32), d])


def apply_rope(x, cos, sin):
    T = x.shape[1]
    half = x.shape[-1] // 2
    c = cos[:T][None]
    s = sin[:T][None]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


TK_HDR = "<IIIIIBBH"
TK_REC = "<fBH"
META = "▁"
TK_CONTROL, TK_USER_DEFINED, TK_BYTE, TK_UNKNOWN = 2, 3, 4, 1


class Tokenizer:
    def __init__(self, blob):
        n, pad, eos, bos, unk, dummy, bfb, _ = struct.unpack_from(TK_HDR, blob, 0)
        off = struct.calcsize(TK_HDR)
        rec = struct.calcsize(TK_REC)
        self.pieces, self.scores, self.types = [], [], []
        for _ in range(n):
            score, t, ln = struct.unpack_from(TK_REC, blob, off)
            off += rec
            self.pieces.append(blob[off:off + ln].decode("utf-8"))
            self.scores.append(score)
            self.types.append(t)
            off += ln
        self.pad, self.eos, self.bos, self.unk = pad, eos, bos, unk
        self.add_dummy, self.byte_fallback = dummy, bfb
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i for i, (p, t)
                        in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types)
                               if t == TK_USER_DEFINED), key=len, reverse=True)

    def _bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best, bj = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best is None or self.scores[idx] > best):
                    best, bj = self.scores[idx], j
            if bj < 0:
                break
            syms[bj:bj + 2] = [syms[bj] + syms[bj + 1]]
        ids = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk)
        return ids

    def encode(self, text):
        if not text:
            return []
        esc = text.replace(" ", META)
        if self.add_dummy:
            esc = META + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            m = next((m for m in self.markers if esc.startswith(m, i)), None)
            if m is not None:
                ids += self._bpe("".join(buf)); buf = []
                ids.append(self.p2id[m]); i += len(m)
            else:
                buf.append(esc[i]); i += 1
        return ids + self._bpe("".join(buf))

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(META, " ")
        return text[1:] if self.add_dummy and text.startswith(" ") else text


def build_prompt(query, tools_json, system=""):
    prefix = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return (prefix + "<|im_start|>user\n<tools>" + tools_json + "</tools>\n"
            + query + "<|im_end|>\n<|im_start|>assistant\n")


DEFAULT_TOOLS = ('[{"name":"get_weather","parameters":{"type":"object",'
                 '"properties":{"city":{"type":"string"}},"required":["city"]}}]')


def load(path):
    """The model and its tokenizer from one `.cact` path."""
    m = Model(path)
    return m, Tokenizer(m.t(m.c.h["num_tensors"] - 1))


def greedy(m, tok, ids, max_new=32):
    """Greedy continuation ids, stopping at EOS."""
    ids = list(ids)
    out = []
    for _ in range(max_new):
        nxt = int(np.argmax(m.forward(np.array(ids, np.int64))[-1]))
        if nxt == tok.eos:
            break
        out.append(nxt)
        ids.append(nxt)
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cact", nargs="?", default="weights/needle3.cact")
    ap.add_argument("query", nargs="?", default="What's the weather in Paris?")
    ap.add_argument("--tools", default=DEFAULT_TOOLS,
                    help="tool schemas as compact JSON")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--expect", default=None,
                    help="exit 1 unless the answer contains this text")
    args = ap.parse_args(argv)

    model, tokenizer = load(args.cact)
    prompt_ids = [tokenizer.bos] + tokenizer.encode(
        build_prompt(args.query, args.tools))
    answer = tokenizer.decode(
        greedy(model, tokenizer, prompt_ids, args.max_new))
    print(answer)
    if args.expect is not None and args.expect not in answer:
        print(f"expected {args.expect!r} in the answer", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
