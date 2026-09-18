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

## What landed in 0.3.0

All verified against `needle3.cact` unless noted. 322 tests across the
workspace; `clippy --all-targets -D warnings` clean.

- **Container** — `needle-infer::cact::{CactV3, CactV3Geometry}`. The header is
  49 u32-sized fields; the tag is one greater than v2's, so a container states
  its own generation in its first word and dispatch never guesses from a file
  name. Checked field for field against `export.read_export`, including all 581
  directory records and their offsets.
- **Canon** — `tools/cact_params_v3.py` inverts `export._tensors` and
  `tools/check_v3_canon.py` runs upstream's own model on the result, which
  produces correct tool calls. That is the check per-tensor parity cannot do:
  the directory is nameless, so a correct tensor in the wrong slot passes every
  individual comparison. `V3Layout` walks the same order in Rust and rejects a
  container whose shapes disagree with its header.
- **Tokenizer** — the embedded blob decodes with the existing `sp_tokenizer`
  unchanged and matches `sentencepiece` exactly on 14 cases.
- **Forward pass** — `needle-core::v3`. Token-exact: 9.0e-6 relative on the
  logits with zero argmax mismatches over 57 positions. Components verified
  separately — HadamardMLP 5.4e-6, attention 2.7e-6, Engram keys/values 1.2e-6
  with the hash indices matching *exactly* as integers.
- **KV cache** — incremental decode is bit-identical to prefill (0.000e0), and
  sized to the session: 1.2 MB for a 57-token run against 14.4 MB if every
  window were reserved.
- **Batched prefill and threading** — 477ms to 238ms over 57 positions, 2.00x,
  bit-identical because `matmul_rows_prepared` matches repeated matvecs exactly.
- **Confidence head** — 2e-6 against the reference. Cells checked first.
- **Constrained decoding** — the JSON state machine over the v3 token table,
  engaged only between the `<tool_call>` markers.
- **Bindings** — Rust, C ABI (12 entry points, header compiled against the
  built library), WASM (`NeedleV3Wasm`), Python (`V3Engine`). Each refuses a v2
  container rather than misreading its header.
- **CLI** — dispatches on the container tag; all three generations from one
  binary.

## Findings worth keeping

- **v3's tokenizer is not v2's.** The two `tokenizer.model` files differ
  (126,520 against 132,396 bytes). Sharing a vocabulary between generations
  would tokenize plausibly and decode to nonsense.
- **"HadamardMLP" contains no Hadamard transform in v3.** The three stages are
  Kronecker products of learned 32x32 factors — measured drift from their Walsh
  initialisation is 1.45 to 3.98 against entries of +/-1. v2's FWHT shortcut
  does not carry over. The Kronecker structure is still the win: 65,536 MACs
  per stage against ~1.05M dense, in L1-resident blocks.
- **Parity must be measured against a float32 reference.** The reference config
  ships bfloat16. Measured against it, a *correct* MLP kernel reads as 9.7
  relative error and a correct forward pass as 8.3e-2 with an argmax mismatch.
  Component fixtures are generated in f32 and deviation is measured against
  output RMS; behavioural parity is the separate token-exact gate. Conflating
  the two sends you hunting a bug that is not there — twice, here.
- **v3 introduces FP32 records**, which v2's container never carried: the two
  Hadamard permutations, stored rather than derived because upstream generates
  them from a seeded RNG a runtime cannot reproduce.
- **`gate_proj` is not v2's per-head gate.** It is `Dense(num_heads *
  v_head_dim)`, an elementwise sigmoid over the attention output.
- **The confidence head is easier to misuse on v3.** It scores a completion,
  not a query. On v2 a bare query scored near zero, so the mistake announced
  itself; on v3 it scores 0.80 against 0.93 for a correct completion and 0.26
  for a wrong one.
- **The engram table count is `orders x heads`** — 2 x 3 = 6, where upstream
  derives `heads = d_model / (len(orders) * 128)`. `engram_geometry`'s 3 is the
  head count, not the table count.
- **Upstream now ships a WebAssembly component** — `needle.component.wasm`
  (4.2 MB) with a real `needle.wit` declaring `cactus:needle@3.0.0`. Their
  native C API remains a four-function global singleton with no handles.

## Still open

- **Decode is ~8.3 ms/token and threading does not help it.** Single-position
  matvecs are too small to pay the rayon dispatch, the same finding v2 reached.
  Batching is a prefill technique. Further gains would come from the CQ group
  decode, which is the remaining bottleneck, not from the FMA.
## The intelligence ladder

An earlier note here said elastic depth was out of scope because `ladder_slice`
was never called. That was wrong about the call graph: `finetune.rung()` calls
`ladder_slice` + `ladder_config`, and `needle build --layers N` ("export the
N-layer rung of the base (2..20)") goes through it. The conclusion that the
container bakes one configuration was right, and is the important part.

**A rung is a physically separate file, not a flag.** `needle build --layers 8`
rewrites the container: `num_layers = 8`, remapped global-attention and Engram
layers, and only the surviving blocks' tensors. The header carries no ladder
metadata — no depth field, no subset, no `ladder_order` — so nothing in the file
records that it was sliced. Cactus's format post puts it plainly: "nothing in
the binary knows or cares which depth it was handed."

**Which blocks survive is bisection, not truncation.** `_ladder_layer_order`
starts from both endpoints and repeatedly splits the widest gap, giving for 20
blocks:

```
0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18
```

Depth *d* keeps the first *d*, sorted. Blocks 0 and 19 are in every rung. Since
block 19 carries both a global-attention layer and an Engram site, no rung is
left without either; all four global layers are present from depth 6.

**We already load rungs — verified on real ones.** Our geometry is entirely
header-driven, so a rung is just a v3 container with different numbers in it.
Two suites cover it. `v3_rung_containers` asserts the shapes across all 19
depths against `tests/v3_ladder_rungs.json`, generated by running upstream's own
`ladder_layer_indices` — deriving the subsets ourselves would risk agreeing with
our own bug. `v3_rung_e2e` then runs containers actually built from upstream's
checkpoint by `tools/build_rung.py`, which calls upstream's `rung()` and
exporter rather than reimplementing either. No loader change was needed at all.

Built and checked at depths 2, 4, 6, 8, 12 and 16. The geometry came out exactly
as `ladder_config` predicts — 8L gives global layers `(1, 3, 5, 7)` and Engram
sites `(4, 7)` — and 6L, 8L, 12L and 16L each produce the right tool call. The
KV cache falls out of `num_layers`: 2.6 MB at 6 layers and 3.5 MB at 8 against
8.8 MB at 20, for a 512-token session. Batched prefill stays bit-identical to
stepped decode at 8 layers, the same invariant the 20-layer suite asserts.

**Shallow rungs degrade, and it is not our code.** At 2 layers the output is
repetitive babble and at 4 there is no tool call at all. Running the independent
numpy reference over the *same container* reproduces it — at 2 layers it
degenerates the same way, while at 6 layers it agrees with our engine
token for token, reasoning text included. Two implementations agreeing puts the
weakness in the weights, not in a depth-dependent bug. Cactus says as much: at
two layers the confidence head "withholds almost every call".

**The public exporter cannot reproduce the published archive.** `needle3.cact`
stores 115 CQ tensors at 2 bits and 7 at 4 bits — embedding, the three
`mhc.phi` tensors and the three head matrices — the `embedding=4, mhc=4,
default=2` policy. The vendored exporter takes a single scalar `bits`, and
`_cq_pack` *raises* unless it equals `WEIGHT_BITS`, which is 4. So
`needle build --layers N` from the public repo emits a uniformly 4-bit rung,
which is larger than the shipped 20-layer file: our 16-layer build is 51.5 MB
against the published 35.3 MB, and the blog's "8 MB at 4 layers to 29 MB at 20
in CQ2-bit" describes files the public tool does not produce. This does not
affect us — the runtime reads the bit width per record — but it means a locally
built rung is not byte-comparable to one Cactus ships, and per-rung parity must
compare the reference against the same container under test, which is how our
suites already work.

**What a load-time slicer would have to do**, if we implement
`load_with_depth` against the 20-layer container: keep the selected blocks'
tensors; keep whole Engram tables for sites whose block survives, renumbered
contiguously; and slice the confidence head's rows `(0, *(layer + 1 for layer
in selected))` from `probes` (axis 0), `gain` (axis 0) and `row_bias` (axis 1),
leaving `query`, `proj` and `bias` alone. There is no adapter to merge — the
blog's `θd = slice(θ + BA, Sd)` describes training; `ladder_slice` operates on
the params directly. The obstacle is packing: `probes`, `query` and `proj` are
Cactus-Quants in the shipped container, so slicing rows means re-packing
quantised data, and our head loader requires CQ for exactly those three.

**mHC lanes move when you slice.** The lane is `eye(n)[arange(L) % n]` — a
function of position in the stack, not of block identity — so block 4 sits in
lane 0 at depth 20 and lane 1 at depth 8. `ladder_slice` does not compensate;
the ladder is trained across sampled depths so that this is fine. Our
`active_lane(layer) = layer % mhc_lanes` recomputes from the new config and is
therefore correct by construction, but an implementation that tried to preserve
the parent's lane assignment would be wrong.

**The confidence head is sliced, not recalibrated.** One head trained at full
depth, with rows taken per surviving block. No per-depth calibration tensor
exists. Cactus notes it grows conservative at shallow depths — at 2 layers it
"withholds almost every call" — which is behaviour, not miscalibration.

**Width slicing remains out of scope**: `width_slice` really is uncalled by any
export path, and the header carries no width field.
- **No contrastive head in the published v3 weights**, so `retrieve_tools` and
  `encode_contrastive` have no v3 equivalent on any surface. This is a property
  of the checkpoint, not the architecture: upstream defines three heads —
  embedding (1), confidence (2), router (3) — and `export.py` emits whichever
  are present in the params, so the released checkpoint simply has no
  `embedding_head`. The container's `heads.manifest` is the single code `2`.
  Were a container to carry code 1 it would load unchanged — `EmbeddingHead`
  subclasses `ProbeHead` and does not override `export`, so it emits the same
  six tensors the canon walk already expects, and only `RouterHead` adds a
  seventh. The embedding would then be this head's forward followed by an L2
  normalisation, its width read from the projection rather than the header.
  Cactus lists text embedding as a Needle 3 capability; that capability is not
  reachable from these weights.

## The int8 KV cache

The container declares `kv_bits = 8`, and upstream maps `>= 8` to
`a8_fake_quant_kv` — `fake_quant(x, x.shape[-1], 8)`, per-head symmetric int8.
Below 8 it maps to `cq_fake_quant_kv` at group 64, a *different* scheme the
shipped checkpoint was not post-trained for, so only the declared width is
implemented. `KvPrecision::Int8` is opt-in; `F32` stays the default because it
is the path verified bit-identical against the reference.

Measured on the shipped checkpoint (`v3_kv_int8`):

| session | f32 cache | int8 cache |
|---|---|---|
| 512 tokens | 9.0 MB | 2.6 MB |
| 2048 tokens | 21.3 MB | 5.9 MB |
| 8192 tokens (full context) | 42.3 MB | 11.5 MB |

27.2% of f32, not 25%: each stored head vector also carries an `f32` scale.
`V3Config::kv_bytes_at` reports this and is asserted against the real
allocation, so the figure a caller budgets from cannot drift from what is
allocated. All three test queries produce byte-identical tool calls.

Two things here were easy to get wrong and are worth keeping:

- **Upstream quantises the query too.** `maybe_quant_query` sits beside
  `maybe_quant_kv` in `architecture.py`, both gated on the same `quant` flag,
  both applied after RoPE and before attention. A KV-only implementation is not
  the numerics the model was post-trained for.
- **Both paths must read the same representation.** The first version attended
  over dequantised `f32` in batched prefill and over stored integers in decode.
  That is algebraically identical and numerically is not — the integer reader
  hoists the per-head scale out of the dot product — and it put 3.7e-2 of drift
  between a prefilled session and a stepped one, against an f32 baseline of
  exactly zero. Both paths now attend through `KvStore`, and
  `int8_prefill_agrees_with_int8_stepping` asserts bit-identity rather than a
  tolerance. Note that the *continuation* test could not see this: the model
  picked the same 24 tokens either way. Only the logits showed it.

The confidence head runs at f32 even inside an int8 session, deliberately.
`quant_rows` fires only when a cache is present, and `forward_head` passes
none — the head pools over its own forward rather than the cached decode loop.
Upstream's `quant` flag is global and would cover it, so a caller comparing a
confidence number against the tokens it scores is comparing two numerical
paths. Left as is because the head is not part of the decode loop the cache
serves, and because the f32 head is the one verified to 2e-6 against the
reference.

## Parity-test audit (v3 against v2)

Done by diffing the two suites' test-function names rather than from memory.
Gaps found and closed:

- **Statelessness.** v2 carried four tests (`reset_yields_identical_second_run`,
  `model_is_stateless_across_sequences`, `shared_state_matches_fresh_state`,
  `head_runs_do_not_affect_generation`); v3 had none, despite holding more
  per-session state than v2 — three histories in `V3Cache` plus a pooled probe
  head. Now `v3_statelessness`, five tests.
- **Tokenizer id range.** v2's `every_encoded_id_is_in_range` had no v3
  counterpart. It matters more in v3 because `clamp_token` would absorb an
  out-of-range id silently. Added to `tokenizer_v3_parity`.
- **The int8 path**, above: quantiser, stored-integer reader, memory, behaviour
  and prefill/decode agreement.
- **CI coverage.** `v3_kv_quant_parity`, `v3_kv_int8` and `v3_statelessness`
  now run in the `parity-v3` job, the first from tracked vectors needing no
  container. `v3_kv_int8` also runs threaded, since its exact-equality
  assertion is what a row-splitting bug breaks.
- **The Python bindings were never tested in CI at all.** `test_v3.py` existed
  and no workflow ran it, so the wheel went to PyPI on the strength of the Rust
  tests alone. The `parity-v3` job now builds the wheel with maturin from the
  repo root — which is where `pyproject.toml` lives and where the distribution
  is named `needle-rs`; building the crate manifest directly produces a
  differently named wheel that is not what ships — installs it, and runs the
  script. It needs `needle2.cact` as well, because the test asserts all three
  generations coexist in one module.
- **Int8 ring growth and wrap.** The int8 continuation test was originally
  hinted generously enough that `ensure` never grew the ring, leaving the
  resize of the four int8 buffers — including the per-head scale arrays, whose
  indexing depends on `slots` — with no coverage. The hint now matches the f32
  sibling's, the test asserts the ring actually grew, and
  `an_int8_ring_wraps_and_grows_like_the_f32_one` covers wrap-around readback
  at the cache level.
- **Int8 behavioural comparison covered only happy paths.** Three prompts that
  should all produce a call cannot see the failure that matters — drift that
  starts *inventing* a call on an unrelated query. Now eight prompts across
  two tools, including two abstentions and a two-call case, asserting both the
  payload and whether a tool was called at all.

Deliberately absent, with reasons:

- **No chunked-prefill invariance test** (v2's `result_is_independent_of_
  prefill_chunk`, `chunk_boundary_does_not_change_the_result`): v3 prefills the
  whole prompt in one pass and the CLI rejects `--prefill-chunk` for v3. There
  is no chunk boundary to be invariant to.
- **No batch-scratch test** (v2's `batch_scratch_is_reusable`): v3 exposes no
  batch API.
- **No contrastive/retrieval tests**: the published v3 weights export only a confidence head.
- **`v3_component_parity` and `v3_forward_parity` are not run in CI** because
  their fixture ladders are gitignored for size; both skip cleanly. CI exercises
  the container, tokenizer, e2e, FFI, KV-quantisation and statelessness suites,
  all from tracked fixtures or the downloaded container.

### Regenerating the ladder rung fixture

```bash
python3 - <<'EOF'
import sys, json; sys.path.insert(0, "needle")
from needle.model.architecture import ladder_layer_indices, _ladder_layer_order
L, GLOBAL, ENGRAM = 20, (4,9,14,19), (3,7,11,15,19)
rows = []
for d in range(2, L+1):
    sel = ladder_layer_indices(L, d); remap = {l:i for i,l in enumerate(sel)}
    rows.append({"depth": d, "selected": list(sel),
                 "global_layers": [remap[l] for l in GLOBAL if l in remap],
                 "engram_sites":  [remap[l] for l in ENGRAM if l in remap],
                 "head_rows": [0] + [l+1 for l in sel]})
json.dump({"parent_layers": L, "ladder_order": list(_ladder_layer_order(L)),
           "parent_global": list(GLOBAL), "parent_engram_sites": list(ENGRAM),
           "rungs": rows}, open("tests/v3_ladder_rungs.json","w"), indent=1)
EOF
```

Needs only a `needle` checkout — no JAX, no checkpoint.

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
| `tests/v3_component_vectors.json` + `.f32` (12 MB) | **no** | `tools/gen_v3_component_parity.py` |
| `tests/v3_forward_vectors.json` + `.f32` (7 MB) | **no** | `tools/gen_v3_forward_parity.py` |

The two large ladders are gitignored on size, as the v2 one is. The canon check
itself is `tools/check_v3_canon.py`, which needs no fixture — it runs upstream's
model on the rebuilt tree and prints what it produces.

Every parity test skips with a printed notice when its inputs are absent, so a
fresh clone runs `cargo test` clean, and CI runs both suites against a freshly
downloaded container. Two checks do run on a fresh clone with nothing
downloaded: `tests/v3_container_offline.rs`, which assembles a synthetic v3
container byte for byte and walks the canon on it, and the header tests in
`cact.rs`.

### The canon check without JAX

`tools/needle3_reference.py` answers the same question as
`tools/check_v3_canon.py` and needs numpy alone. It reads the container
directly — geometry, weights, permutations and tokenizer all ride inside it —
so it needs neither the parity interpreter nor a pinned `needle` checkout:

```
python3 tools/needle3_reference.py weights/needle3.cact --expect get_weather
```

It recomputes every position from the whole prefix, so it takes about three
minutes for one answer. The JAX oracles stay the primary reference, because
they are upstream's own code; this one is a second opinion for a contributor
who cannot install JAX, and an independent transcription of
`architecture.py` for anyone reading the port.

## Reference files (upstream, treat as spec)

Vendored at `needle/`, pinned to the local branch `needle3-oracle` at
`dd85774 Needle 3 Live`. The tracked fixtures were generated against that
commit, so moving the clone changes the oracle underneath them — check out
`needle3-oracle` rather than pulling, and regenerate if you advance it.

- `needle/model/export.py` — `.cact` byte layout, `_tensors` canon order, `read_export`
- `needle/model/architecture.py` — the model; `head_dims`, `engram_geometry`, `Stack`
- `needle/model/quantize.py` — Cactus-Quants codebooks
- `needle/model/run.py` — `_get_decode_fn`, `generate`; the parity oracle
