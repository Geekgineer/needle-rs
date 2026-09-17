# needle-rs 0.3.0 — Needle 3 port design

Upstream released **Needle 3** (`Cactus-Compute/needle3`) on 2026-09-17. This
design covers porting it into needle-rs beside v1 and v2, to the same standard
the v2 port set: token-exact against the reference, verified in CI, with the
full binding surface.

Companion documents: [ARCHITECTURE.md](../../../ARCHITECTURE.md) for how the
runtime is structured, [docs/v2-port-record.md](../../v2-port-record.md) for the
method this port follows.

## Goal and scope

**Goal.** `needle-rs` 0.3.0 runs Needle 3 from upstream's shipped `needle3.cact`
with no conversion step, reaching the same feature surface v2 has today, and v1
and v2 keep working unchanged.

**In scope** — the full v2-equivalent surface:

- `.cact` v3 container: header, codebook, positional directory, embedded tokenizer
- Forward pass, token-exact against upstream's reference on the shipped weights
- Batched prefill, `parallel` (rayon) threading, streaming, temperature sampling
- Constrained JSON decoding over the v3 token table
- The confidence head
- All four bindings: Rust, C ABI, WASM, Python
- Parity gates in CI, on the same pattern as `parity-v2`

**Explicitly out of scope**, with the evidence for each:

- **Ladder (elastic depth) and width slicing.** `ladder_slice`, `width_slice` and
  `ladder_row_keep` appear only in `needle/model/architecture.py`; `export.py`,
  `run.py` and `checkpoints.py` never call them, and the container header carries
  no depth or width field. The shipped `.cact` bakes one configuration, so a
  container-driven runtime cannot express these and does not need to.
- **A contrastive/retrieval head for v3.** `config.json` declares
  `extras.heads = ["confidence"]`. `RouterHead` and `EmbeddingHead` exist in the
  reference but are not exported, so `retrieve_tools` has no v3 equivalent. This
  mirrors the v1 situation and must be documented, not silently degraded.
- **Upstream's composed confidence score.** Unchanged from v2: `apis.md` defines
  it as the minimum of the calibration head and the decode probability, and the
  composition lives in the compiled engine with no reference to verify against.
- **Re-hosting weights.** needle-rs reads upstream's container directly.

## Shipped geometry

Read from `weights/needle3.cact` (35,335,380 bytes) with upstream's own
`export.read_export`, not copied from documentation:

```
vocab 8192   out_vocab 8192   d_model 768
heads 12     kv_heads 2       qk_head_dim 48   v_head_dim 64
layers 20    max_seq_len 8192 rope_theta 100000.0
hada_n 1024  mhc_lanes 4      sliding_window 1024
global_layers (4, 9, 14, 19)  qkv_conv_taps 3
engram: slots 18432  sub_dim 128  tables 6  conv_taps 4  dilation 3
        seed_heads 0  orders (2,3)  sites (3,7,11,15,19)
kv_window 256  kv_bits 8   codebook_len 28
581 tensors, 1 RAW (the embedded tokenizer)
121,021,910 parameters
```

## What actually changed from v2

This is the engineering work. Everything else is reuse.

| | v2 | v3 |
|---|---|---|
| Shape | 27 × 512, 8/4 heads, head_dim 64 | 20 × 768, 12/2 heads |
| Head dims | symmetric 64 | **asymmetric: qk 48, v 64** |
| Attention span | single window 256 | **hybrid: sliding 1024, full-attention at layers 4/9/14/19** |
| QKV | linear projections | **+ depthwise short convolution, 3 taps** |
| Engram | 2 sites (2,15), 8192 slots, 4 tables | **5 sites (3,7,11,15,19), 18432 slots, 6 tables, seed_heads** |
| `hada_n` | 512 | 1024 |
| Context | 2048 | 8192 |
| Exported heads | contrastive + confidence | **confidence only** |
| Container header | 120 bytes | **49 u32-sized fields, 64-bit global-layer bitmask, 16 site slots** |
| License | MIT | **Apache-2.0** |

Unchanged and therefore reusable: Cactus-Quants (same codebook scheme,
`embedding=4,mhc=4,default=2`, group 128), the FWHT, mHC with Sinkhorn lane
mixing, HadamardMLP, ZCRMSNorm on q/k, the sigmoid `gate_proj`, RoPE, and the
`.cact` container's overall layout (header → codebook → positional directory →
64-byte-aligned blobs).

## Approach: new assembly, shared primitives

Considered and rejected:

- **Mirror the v2 port wholesale.** Lowest risk, but copies the attention block,
  mHC mixing, Engram and HadamardMLP into a third near-identical implementation
  that then diverges. Every future optimisation would be done three times.
- **Unify v2 and v3 into one geometry-parameterised decoder.** Attractive on
  paper — they share most primitives — but it rewrites a forward pass currently
  verified token-exact against 788 capture points, trading a proven path for
  reduced duplication, and breaks the additive rule the v2 port was held to.

**Chosen: extract only provably identical primitives, write new code for
everything v3 does differently.** The v2 forward pass is not edited, so
`v2_e2e_parity` stays green by construction rather than by re-verification.

The shared layer already exists and works — v2 consumes `crate::cq`,
`crate::hadamard`, `crate::norm::zc_rms_norm_vec`, `crate::ops::{sigmoid,
softmax_inplace}`, `crate::rope::RopeCache` and `crate::math`. This port extends
it by one step.

**The bright line for promotion:** a function moves to the shared layer only if
v2 and v3 call it with identical semantics and it bakes in no geometry. Anything
conditional on generation stays duplicated. Applying that test to
`v2/kernels.rs`:

| Function | Decision |
|---|---|
| `rms_unit`, `rms_unit_to` | **promote** — pure, no geometry |
| `sinkhorn` | **promote** — operates on an `n × n` matrix passed in |
| `silu` | **promote** — scalar |
| `hadamard_mlp` | **promote** — `hada_n` is already an argument, not a constant |
| `engram_index`, `engram_table_order` | **keep separate** — v3 adds seed-heads and a different table/order mapping |

v2 keeps its current import paths through re-exports, so the promotion is a move
plus a `pub use`, with no change to v2 call sites.

### Module plan

New, mirroring `v2/`:

```
crates/needle-core/src/
  kernels.rs            promoted from v2/kernels.rs (rms_unit, sinkhorn, silu, hadamard_mlp)
  v3/mod.rs
  v3/config.rs          V3Config from the 49-field header
  v3/model.rs           forward pass: conv-QKV, asymmetric GQA, hybrid mask, 5-site Engram
  v3/kernels.rs         v3-only: engram indexing with seed-heads, short conv
  v3/batch.rs           batched prefill
  v3/heads.rs           confidence head
crates/needle-infer/src/
  cact.rs               +v3 header branch, version-dispatched (existing file)
  v3_engine.rs          prompt template, sampling, streaming, tool-call extraction
  v3.rs                 public re-export surface
```

Bindings extend existing files: `needle-c/src/v3.rs` + header declarations,
`needle-wasm/src/v3.rs` (`NeedleV3Wasm`), `needle-python` (`V3Engine`).

### Container dispatch

`cact.rs` currently parses one header shape. It gains a version discriminator
read from the tag/geometry, dispatching to a v2 or v3 header reader and yielding
a generation-tagged model. A container whose header does not match either layout
fails at load rather than producing wrong logits — the existing v2 rule.

## Parity strategy

The method that made v2 trustworthy, applied unchanged. Per-tensor comparison
cannot catch a canon error, because the directory is nameless and a correct
tensor in the wrong positional slot passes every individual check. So the canon
is pinned by inverting the export and running upstream's own code on the result.

`tools/cact_params_v3.py` inverts `export._tensors` to rebuild the Flax parameter
tree straight from `needle3.cact`, so upstream's reference decode runs on the
shipped weights. Verified feasible: `export.read_export` parses the shipped
container, and `TransformerConfig`'s defaults already agree with it
(`head_dims → (48,64)`, `qkv_conv_taps 3`, `sliding_window 1024`,
`global_layers (4,9,14,19)`).

Fixture ladder, mirroring v2's:

| Fixture | Generator | Tracked |
|---|---|---|
| `tests/cact_v3_vectors.json` | `tools/gen_cact_v3_parity.py` | yes |
| `tests/tokenizer_v3_vectors.json` | `tools/gen_tokenizer_v3_parity.py` | yes |
| `tests/v3_forward_vectors.json` + `.f32` | `tools/gen_v3_forward_parity.py` | no (size) |
| `tests/v3_heads_vectors.json` | `tools/gen_v3_heads_parity.py` | yes |
| `tests/v3_e2e_vectors.json` | `tools/gen_v3_e2e_parity.py` | yes |

Every parity test skips with a printed notice when its inputs are absent, so a
fresh clone runs `cargo test` clean. CI gains a `parity-v3` job on the
`parity-v2` pattern, downloading the container from HuggingFace and running the
suites plus the threaded configuration, since row-splitting must stay
bit-identical.

**The acceptance gate:** token-exact greedy continuation over a prompt/tool
corpus, matching the reference on the shipped weights. Nothing ships until that
passes, both with and without `parallel`.

## Memory and size

These change the browser story and must be measured, not estimated, before any
public claim:

- Container is 35.3 MB against v2's 13.7 MB — 2.6× the download for a browser
  deployment.
- The KV ring is sized to `kv_window` (256) as in v2, but the geometry differs:
  20 layers × 2 kv_heads × (48 + 64). The global-attention layers at 4/9/14/19
  attend across the full sequence, so they cannot use the 256-slot ring and need
  their own allocation up to `sliding_window`/sequence length. **This is the one
  place the v2 memory design does not transfer, and it needs its own decision
  during implementation rather than an assumption here.**
- The WASM module grows by a third engine. The published figure is currently
  "413 KB module, 156 KB over the wire"; it will be re-measured and re-stated,
  including in `assets/banner.svg`, which bakes the number into the graphic.

## Risks

1. **Global-attention layers break the fixed KV ring.** Highest-impact unknown.
   Mitigation: decide the allocation strategy before writing the cache, and gate
   it on a parity test that includes a prompt longer than the sliding window.
2. **Engram table count is ambiguous.** The container reports
   `num_engram_tables 6` while `engram_geometry` returns 3 — probably 3 tables ×
   2 orders, but a wrong mapping produces plausible-looking wrong logits.
   Mitigation: assert the relationship against the reference before building on it.
3. **Reference cost at 121M.** The v2 forward ladder was generated on CPU JAX at
   45M. At 121M with 8192 context the fixture generation may be slow enough to
   need a reduced capture set. Mitigation: measure early; the ladder is already
   gitignored for size, so a smaller capture set is acceptable if the e2e gate
   holds.
4. **Apache-2.0.** v3 is Apache-2.0 where v1/v2 were MIT. The runtime stays MIT,
   but `CITATION.cff`, the HF model card and any blanket "MIT throughout" claim
   need per-generation wording, and Apache-2.0 carries attribution/NOTICE
   obligations MIT does not.

## Milestones

Each is independently verifiable, in dependency order:

1. **Container** — v3 header parse, canon derived from geometry, every slot's
   shape validated. Gate: matches `read_export` field for field.
2. **Tokenizer** — embedded SP-BPE. Gate: exact token-ID match on a corpus.
3. **Forward pass** — conv-QKV, asymmetric GQA, hybrid mask, 5-site Engram,
   mHC, HadamardMLP, tied LM head. Gate: capture points match the reference.
4. **Engine and CLI** — prompt template, sampling, streaming, extraction.
   Gate: token-exact greedy continuation.
5. **Performance** — batched prefill, threading. Gate: bit-identical to sequential.
6. **Heads and constrained decoding.** Gate: confidence matches the reference.
7. **Bindings** — C ABI, WASM, Python. Gate: smoke tests on each surface.
8. **Docs and release** — measured sizes everywhere including the banner, the
   license note, and a `parity-v3` CI job.
