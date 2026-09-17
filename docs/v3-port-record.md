# needle-rs 0.3.0 — Needle 3 port record

Upstream released **Needle 3** on 2026-09-17. needle-rs 0.2.x runs v1 and v2;
0.3.0 adds v3 alongside them. This is the running record of that port — what
changed upstream, what the shipped container actually declares, and what has
been verified so far.

Design and scope: [superpowers/specs/2026-09-17-needle-v3-port-design.md](superpowers/specs/2026-09-17-needle-v3-port-design.md).
The v2 port's record, whose method this follows: [v2-port-record.md](v2-port-record.md).

## What changed upstream

| | v2 (45M) | v3 (121M) |
|---|---|---|
| Shape | 27 × 512, 8/4 heads, head_dim 64 | 20 × 768, 12/2 heads |
| Head dims | symmetric 64 | **asymmetric: qk 48, v 64** |
| Attention | one sliding window of 256 | **hybrid: sliding 1024, full attention at layers 4/9/14/19** |
| QKV | linear projections | **+ causal depthwise conv, 3 taps** |
| Engram | 2 sites, 8192 slots, 4 tables | **5 sites (3,7,11,15,19), 18432 slots, 6 tables, seed heads** |
| `hada_n` | 512 | 1024 |
| Context | 2048 | 8192 |
| Exported heads | contrastive + confidence | **confidence only** |
| Container header | 120 bytes, tag `0x05E12A83` | **196 bytes, tag `0x05E12A84`** |
| Container size | 13.7 MB | **35.3 MB** |
| Licence | MIT | **Apache-2.0** |

Unchanged, and therefore reused rather than rewritten: Cactus-Quants (same
`embedding=4,mhc=4,default=2` scheme at group 128), the FWHT, mHC with Sinkhorn
lane mixing, HadamardMLP, ZCRMSNorm on q/k, RoPE, the 44-byte directory record,
the codebook layout, 64-byte blob alignment, and the SentencePiece blob
*format* — though not the model itself, which differs; see Findings.

## Shipped geometry

Read from `weights/needle3.cact` (35,335,380 bytes) with upstream's own
`export.read_export`, never copied from documentation:

```
vocab 8192   out_vocab 8192   d_model 768
heads 12     kv_heads 2       qk_head_dim 48   v_head_dim 64
layers 20    max_seq_len 8192 rope_theta 100000.0
hada_n 1024  mhc_lanes 4      sliding_window 1024
global_layers (4, 9, 14, 19)  qkv_conv_taps 3
engram: slots 18432  sub_dim 128  tables 6  conv_taps 4  dilation 3
        seed_heads 0  orders (2,3)  sites (3,7,11,15,19)
kv_window 256  kv_bits 8   codebook_len 28
581 tensors: 122 CQ, 456 FP16, 2 FP32, 1 RAW (the embedded tokenizer)
121,021,910 parameters
```

## What is done

- **Container** — `needle-infer::cact::{CactV3, CactV3Geometry}`. The header is
  49 u32-sized fields; the tag is one greater than v2's, so a container states
  its own generation in the first word and dispatch never guesses from length.
  A v2 loader rejects a v3 tag outright rather than reading 196 bytes as 120 and
  producing plausible wrong geometry.
- **Shared body parsing** — `parse_directory` and `parse_codebook` are now free
  functions both generations call. The record layout and codebook are identical;
  only the header ahead of them changed.
- **Container parity** — checked field for field against `export.read_export`:
  geometry, all 581 directory records including offsets and byte counts, and the
  container's own size. `tools/gen_cact_v3_parity.py` generates the fixture.
- **Tokenizer** — the container's embedded blob decodes with the existing
  `sp_tokenizer` unchanged, and matches real `sentencepiece` exactly on 14 cases.
- **Canon pinned** — `tools/cact_params_v3.py` inverts `export._tensors` to
  rebuild the Flax tree from the container, and `tools/check_v3_canon.py` runs
  upstream's own `SimpleAttentionNetwork` on it. Both shipped prompts produce
  correct tool calls, which is the evidence per-tensor parity cannot give.
- **Canon in Rust** — `needle-infer::v3::V3Layout` walks the same order and
  validates every slot whose shape the geometry determines. A container
  claiming one layer fewer is rejected rather than silently misread.
- **Core geometry** — `needle-core::v3::V3Config`, including the KV design:
  `attention_span()` distinguishes the 16 sliding layers from the 4 global
  ones, and `kv_bytes()` sizes a session to the sequence it actually uses.
  2.2 MB at int8 for 512 tokens against 10.5 MB if the global layers were
  preallocated at `max_seq_len`.
- **Forward ladder captured** — `tools/gen_v3_forward_parity.py` records input
  embeddings, RoPE tables, engram keys/values at all five sites, and logits, so
  a Rust mismatch localises to a component. One reference forward is 1.2 s.

## Findings worth keeping

- **v3's tokenizer is not v2's.** The two `tokenizer.model` files differ
  (126,520 against 132,396 bytes). Reusing the v2 vocabulary for v3 would
  tokenize plausibly and decode to nonsense, so the vocabularies must never be
  shared between generations.
- **v3 introduces FP32 records**, which v2's container never carried: two
  vectors of length 1024 (= `hada_n`) at slots 550–551. The dtype mix is
  asserted so a silent change to the quantisation scheme fails at load.
- **The special-token block grew.** Beyond v2's chat and tool markers, v3 adds
  `<think>`, `<extract>`, `<schema>` and modality tokens (`<image>`, `<speech>`,
  `<tts>`, `<imagen>`). They must resolve to single ids or prompt construction
  would emit them as literal text.
- **`gate_proj` is not v2's per-head gate.** It is `Dense(num_heads *
  v_head_dim)` — an elementwise sigmoid over the attention output, 768x768 on
  this model rather than 12x768.
- **v3 emits `<think>` chain-of-thought** before tool calls, which v2 did not,
  and generation must stop on `<|im_end|>` rather than EOS alone. Both affect
  the default token budget and the tool-call extractor.
- **Upstream now ships a WebAssembly component** — `wasm-component/needle.component.wasm`
  (4.2 MB) with a real `needle.wit` declaring `cactus:needle@3.0.0`. Their
  native C API remains a four-function global singleton with no handles.

## Still to do

Milestones 3–8 of the design: the forward pass, engine and CLI, batched prefill
and threading, the confidence head, constrained decoding, the four bindings, and
the release work.

**The forward-pass plan is deliberately not written yet.** It is blocked on one
decision: v3's global-attention layers (4, 9, 14, 19) attend across the whole
sequence, so they cannot use the fixed 256-slot KV ring that v2 relies on. That
allocation strategy has to be settled before the plan can be written without
placeholders, and it is the single highest-impact unknown in the port.

One other open item, recorded so it is not rediscovered:

- **Apache-2.0.** v3 is Apache-2.0 where v1 and v2 were MIT. The runtime stays
  MIT, but `CITATION.cff`, the Hugging Face model card and any blanket "MIT
  throughout" claim need per-generation wording, and Apache-2.0 carries
  attribution obligations MIT does not.

## Regenerating the fixtures

Everything needs the container, which is gitignored:

```
hf download Cactus-Compute/needle3 needle3.cact --local-dir weights/
hf download Cactus-Compute/needle3 tokenizer/tokenizer.model --local-dir weights/v3/
```

The generators need the parity interpreter, since `jax` has no wheels for the
current system Python:

```
uv venv --python 3.12 .venv-parity
uv pip install --python .venv-parity/bin/python "jax[cpu]" flax numpy sentencepiece
```

| Fixture | Tracked | Generator |
|---|---|---|
| `tests/cact_v3_vectors.json` (92 KB) | yes | `tools/gen_cact_v3_parity.py` |
| `tests/tokenizer_v3_vectors.json` (5 KB) | yes | `tools/gen_tokenizer_v3_parity.py` |

Every parity test skips with a printed notice when its inputs are absent, so a
fresh clone runs `cargo test` clean, and CI runs both suites against a freshly
downloaded container.

## Reference files (upstream, treat as spec)

Vendored at `needle/`, pinned to the local branch `needle3-oracle` at
`dd85774 Needle 3 Live`. The tracked fixtures were generated against that
commit, so moving the clone changes the oracle underneath them — check out
`needle3-oracle` rather than pulling, and regenerate if you advance it.

- `needle/model/export.py` — `.cact` byte layout, `_tensors` canon order, `read_export`
- `needle/model/architecture.py` — the model; `head_dims`, `engram_geometry`, `Stack`
- `needle/model/quantize.py` — Cactus-Quants codebooks
- `needle/model/run.py` — `_get_decode_fn`, `generate`; the parity oracle
