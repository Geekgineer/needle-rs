# Needle 3 Container & Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load `needle3.cact` in needle-rs — header, codebook, directory and embedded tokenizer — verified field-for-field against upstream's own reader, with v1 and v2 untouched.

**Architecture:** The v3 container keeps v2's overall shape (header → codebook → nameless positional directory → 64-byte-aligned blobs) and changes only the header, which grows from 30 to 49 u32-sized fields. The tag differs by one (`0x05E12A84` vs `0x05E12A83`), so `Cact::from_bytes` dispatches on it and returns a generation-tagged value. The 44-byte directory record and the codebook layout are unchanged and are reused as-is.

**Tech Stack:** Rust 2021, MSRV 1.87, no_std-compatible core; Python 3.12 + jax[cpu] + flax in `.venv-parity` for fixture generation; upstream reference vendored at `needle/` (currently `dd85774 Needle 3 Live`).

**Spec:** `docs/superpowers/specs/2026-09-17-needle-v3-port-design.md`

## Global Constraints

- **Additive only.** v1 and v2 code paths are not edited. `cargo test --workspace` must stay green throughout, including `v2_e2e_parity`.
- **MSRV 1.87.** No newer language or std features.
- **`needle-core` is `no_std`-compatible.** No `std` imports in `v3/` without a feature gate; `needle-infer` may use `std`.
- **Clippy clean:** `cargo clippy --workspace --exclude needle-python --all-targets -- -D warnings`.
- **Formatting:** `cargo fmt --all -- --check` must pass before every commit. CI fails on this and has caught it before.
- **Geometry is read, never assumed.** Every figure in code comments or docs comes from the shipped container, not from `config.json` or upstream prose.
- **A wrong container fails at load.** Never parse permissively into wrong logits.
- **v3 header field order (canonical, from `export.py` `_HDR_FMT = "<48If"`):** indices 0..48 =
  `tag, num_tensors, codebook_len, kv_window, kv_bits, vocab, out_vocab, d_model, num_heads, num_kv_heads, num_layers, qk_head_dim, v_head_dim, max_seq_len, hada_n, mhc_lanes, sliding_window, global_mask_lo, global_mask_hi, qkv_conv_taps, engram_slots, engram_sub_dim, num_engram_tables, engram_conv_taps, engram_conv_dilation, engram_seed_heads, num_engram_orders, engram_orders[4] (27..31), num_engram_sites, engram_sites[16] (32..48), rope_theta (48, f32 bits)`.
- **Expected values for the shipped `needle3.cact`** (35,335,380 bytes, 581 tensors): vocab 8192, out_vocab 8192, d_model 768, heads 12, kv_heads 2, layers 20, qk_head_dim 48, v_head_dim 64, max_seq_len 8192, hada_n 1024, mhc_lanes 4, sliding_window 1024, global_layers (4,9,14,19), qkv_conv_taps 3, engram_slots 18432, engram_sub_dim 128, num_engram_tables 6, engram_conv_taps 4, engram_conv_dilation 3, engram_seed_heads 0, orders (2,3), sites (3,7,11,15,19), kv_window 256, kv_bits 8, codebook_len 28, rope_theta 100000.0.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/needle-infer/src/cact.rs` (modify) | Add `TAG_V3`, `HEADER_BYTES_V3`, `CactV3Geometry`, tag dispatch in `from_bytes`. Directory/codebook/blob code is reused unchanged. |
| `crates/needle-infer/tests/cact_v3_parity.rs` (create) | Parity of parsed geometry against the fixture generated from upstream's `read_export`. |
| `tools/gen_cact_v3_parity.py` (create) | Emits `crates/needle-infer/tests/cact_v3_vectors.json` from `needle3.cact` via upstream's reader. |
| `crates/needle-infer/tests/tokenizer_v3_parity.rs` (create) | Confirms the embedded v3 tokenizer blob round-trips through the existing `sp_tokenizer`. |
| `tools/gen_tokenizer_v3_parity.py` (create) | Emits `tokenizer_v3_vectors.json` from upstream's `parse_tokenizer_blob` + sentencepiece. |

---

### Task 1: v3 geometry type and tag dispatch

**Files:**
- Modify: `crates/needle-infer/src/cact.rs`
- Test: `crates/needle-infer/src/cact.rs` (inline `#[cfg(test)]` module, matching the existing pattern at the bottom of the file)

**Interfaces:**
- Consumes: existing `CactError`, `Record`, `REC_BYTES`, `cq::CODEBOOK_LEN`.
- Produces: `pub const TAG_V3: u32`, `pub const HEADER_BYTES_V3: usize`, `pub struct CactV3Geometry { .. }`, `pub enum CactGen { V2(CactGeometry), V3(CactV3Geometry) }`, and `CactV3Geometry::global_layers() -> Vec<usize>`.

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` in `cact.rs`:

```rust
#[test]
fn v3_header_parses_every_field() {
    // 49 u32-sized fields; rope_theta rides in the last one as f32 bits.
    let mut h = vec![0u32; 49];
    h[0] = TAG_V3;
    h[1] = 581;   // num_tensors
    h[2] = 28;    // codebook_len
    h[3] = 256;   // kv_window
    h[4] = 8;     // kv_bits
    h[5] = 8192;  // vocab
    h[6] = 8192;  // out_vocab
    h[7] = 768;   // d_model
    h[8] = 12;    // num_heads
    h[9] = 2;     // num_kv_heads
    h[10] = 20;   // num_layers
    h[11] = 48;   // qk_head_dim
    h[12] = 64;   // v_head_dim
    h[13] = 8192; // max_seq_len
    h[14] = 1024; // hada_n
    h[15] = 4;    // mhc_lanes
    h[16] = 1024; // sliding_window
    h[17] = (1 << 4) | (1 << 9) | (1 << 14) | (1 << 19); // global_mask_lo
    h[18] = 0;    // global_mask_hi
    h[19] = 3;    // qkv_conv_taps
    h[20] = 18432;// engram_slots
    h[21] = 128;  // engram_sub_dim
    h[22] = 6;    // num_engram_tables
    h[23] = 4;    // engram_conv_taps
    h[24] = 3;    // engram_conv_dilation
    h[25] = 0;    // engram_seed_heads
    h[26] = 2;    // num_engram_orders
    h[27] = 2;
    h[28] = 3;
    h[31] = 5;    // num_engram_sites
    h[32] = 3;
    h[33] = 7;
    h[34] = 11;
    h[35] = 15;
    h[36] = 19;
    h[48] = 100000.0f32.to_bits();

    let g = CactV3Geometry::from_words(&h).expect("header should parse");
    assert_eq!(g.d_model, 768);
    assert_eq!(g.num_heads, 12);
    assert_eq!(g.num_kv_heads, 2);
    assert_eq!(g.qk_head_dim, 48);
    assert_eq!(g.v_head_dim, 64);
    assert_eq!(g.num_layers, 20);
    assert_eq!(g.sliding_window, 1024);
    assert_eq!(g.qkv_conv_taps, 3);
    assert_eq!(g.engram_slots, 18432);
    assert_eq!(g.engram_orders, vec![2, 3]);
    assert_eq!(g.engram_sites, vec![3, 7, 11, 15, 19]);
    assert_eq!(g.global_layers(), vec![4, 9, 14, 19]);
    assert_eq!(g.rope_theta, 100000.0);
}

#[test]
fn v3_global_mask_spans_both_words() {
    let mut h = vec![0u32; 49];
    h[0] = TAG_V3;
    h[10] = 40;          // 40 layers, so bits land in the high word too
    h[17] = 1 << 4;      // layer 4
    h[18] = 1 << 3;      // layer 35 = 32 + 3
    let g = CactV3Geometry::from_words(&h).expect("header should parse");
    assert_eq!(g.global_layers(), vec![4, 35]);
}

#[test]
fn v2_tag_is_not_accepted_as_v3() {
    let mut h = vec![0u32; 49];
    h[0] = TAG; // the v2 tag
    assert!(matches!(
        CactV3Geometry::from_words(&h),
        Err(CactError::BadTag(_))
    ));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p needle-infer --lib cact::tests::v3_ -- --nocapture`
Expected: FAIL to compile — `cannot find type CactV3Geometry in this scope`, `cannot find value TAG_V3`.

- [ ] **Step 3: Write the minimal implementation**

Add near the existing `TAG` / `HEADER_BYTES` constants in `cact.rs`:

```rust
/// Needle 3 container tag. One greater than [`TAG`], so a container
/// declares its generation in the first word and dispatch never guesses.
pub const TAG_V3: u32 = 0x05E1_2A84;

/// v3 header: 48 u32 then rope_theta as f32 (`_HDR_FMT = "<48If"` upstream).
pub const HEADER_BYTES_V3: usize = 49 * 4;
```

Then the geometry type:

```rust
/// Geometry declared by a Needle 3 container header.
///
/// Field order is fixed by upstream `export.py`; see `HEADER` there. The
/// runtime derives everything from this, so one binary runs any
/// configuration of the architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct CactV3Geometry {
    pub num_tensors: usize,
    pub codebook_len: usize,
    pub kv_window: usize,
    pub kv_bits: u32,
    pub vocab_size: usize,
    /// Tied text-slice head; 0 means the full vocab. Rows past this are
    /// input-only code embeddings.
    pub out_vocab: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    /// Query/key head width. Differs from `v_head_dim` in v3.
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub hada_n: usize,
    pub mhc_lanes: usize,
    /// Per-layer local attention width. Layers in `global_layers` ignore it.
    pub sliding_window: usize,
    global_mask: u64,
    /// Causal depthwise conv width over Q, K and V. 0 means none.
    pub qkv_conv_taps: usize,
    pub engram_slots: usize,
    pub engram_sub_dim: usize,
    pub num_engram_tables: usize,
    pub engram_conv_taps: usize,
    pub engram_conv_dilation: usize,
    pub engram_seed_heads: usize,
    pub engram_orders: Vec<usize>,
    pub engram_sites: Vec<usize>,
    pub rope_theta: f32,
}

impl CactV3Geometry {
    /// Parse the 49 header words. Caller supplies at least that many.
    pub fn from_words(w: &[u32]) -> Result<Self, CactError> {
        if w.len() < 49 {
            return Err(CactError::TooShort {
                need: HEADER_BYTES_V3,
                got: w.len() * 4,
            });
        }
        if w[0] != TAG_V3 {
            return Err(CactError::BadTag(w[0]));
        }
        let num_orders = w[26] as usize;
        let num_sites = w[31] as usize;
        let take = |base: usize, n: usize, cap: usize| -> Vec<usize> {
            (0..n.min(cap)).map(|k| w[base + k] as usize).collect()
        };
        Ok(Self {
            num_tensors: w[1] as usize,
            codebook_len: w[2] as usize,
            kv_window: w[3] as usize,
            kv_bits: w[4],
            vocab_size: w[5] as usize,
            out_vocab: w[6] as usize,
            d_model: w[7] as usize,
            num_heads: w[8] as usize,
            num_kv_heads: w[9] as usize,
            num_layers: w[10] as usize,
            qk_head_dim: w[11] as usize,
            v_head_dim: w[12] as usize,
            max_seq_len: w[13] as usize,
            hada_n: w[14] as usize,
            mhc_lanes: w[15] as usize,
            sliding_window: w[16] as usize,
            global_mask: (w[17] as u64) | ((w[18] as u64) << 32),
            qkv_conv_taps: w[19] as usize,
            engram_slots: w[20] as usize,
            engram_sub_dim: w[21] as usize,
            num_engram_tables: w[22] as usize,
            engram_conv_taps: w[23] as usize,
            engram_conv_dilation: w[24] as usize,
            engram_seed_heads: w[25] as usize,
            engram_orders: take(27, num_orders, 4),
            engram_sites: take(32, num_sites, 16),
            rope_theta: f32::from_bits(w[48]),
        })
    }

    /// Layers that attend over the whole sequence rather than
    /// `sliding_window`. Bit i of the 64-bit mask marks layer i.
    pub fn global_layers(&self) -> Vec<usize> {
        (0..self.num_layers)
            .filter(|i| self.global_mask >> i & 1 == 1)
            .collect()
    }

    /// True when layer `i` attends globally.
    pub fn is_global(&self, i: usize) -> bool {
        i < 64 && self.global_mask >> i & 1 == 1
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p needle-infer --lib cact::tests::v3_ -- --nocapture`
Expected: PASS, 3 tests.

- [ ] **Step 5: Confirm v1/v2 are untouched**

Run: `cargo test -p needle-infer --lib`
Expected: PASS, no pre-existing test regressions.

- [ ] **Step 6: Format, lint, commit**

```bash
cargo fmt --all
cargo clippy -p needle-infer --all-targets -- -D warnings
git add crates/needle-infer/src/cact.rs
git commit -m "cact: parse the Needle 3 header

49 u32-sized fields against v2's 30. New: out_vocab, split qk/v head
dims, sliding_window, a 64-bit global-attention layer mask, qkv conv
taps, engram seed heads, and up to 16 engram sites.

The v3 tag is one greater than v2's, so a container declares its own
generation and dispatch never has to guess from length."
```

---

### Task 2: Parity fixture from upstream's reader

**Files:**
- Create: `tools/gen_cact_v3_parity.py`
- Create: `crates/needle-infer/tests/cact_v3_parity.rs`

**Interfaces:**
- Consumes: `CactV3Geometry::from_words` and `global_layers()` from Task 1.
- Produces: `crates/needle-infer/tests/cact_v3_vectors.json` with a `geometry` object and a `records` array of `{index, dtype, ndim, shape, group, bits}`.

- [ ] **Step 1: Write the generator**

```python
#!/usr/bin/env python3
"""Emit v3 container parity vectors from upstream's own reader.

Upstream is the spec: whatever `export.read_export` says the container
declares is what needle-rs must agree with, field for field. Run with the
parity venv, which has jax/flax:

    PYTHONPATH=needle .venv-parity/bin/python tools/gen_cact_v3_parity.py
"""
import json
import pathlib
import struct
import sys

from needle.model import export

CONTAINER = pathlib.Path("weights/needle3.cact")
OUT = pathlib.Path("crates/needle-infer/tests/cact_v3_vectors.json")


def main():
    if not CONTAINER.exists():
        sys.exit(f"missing {CONTAINER}; see docs/v2-port-record.md for the download")
    meta, _tensors = export.read_export(str(CONTAINER))

    raw = CONTAINER.read_bytes()
    n_rec = meta["num_tensors"]
    rec_size = struct.calcsize(export._REC_FMT)
    dir_start = struct.calcsize(export._HDR_FMT) + meta["codebook"].size * 4

    records = []
    for i in range(n_rec):
        off = dir_start + i * rec_size
        dtype, ndim, _pad, s0, s1, s2, s3, offset, nbytes, group, bits = (
            struct.unpack_from(export._REC_FMT, raw, off)
        )
        records.append({
            "index": i,
            "dtype": dtype,
            "ndim": ndim,
            "shape": [s0, s1, s2, s3][:ndim],
            "group": group,
            "bits": bits,
        })

    geometry = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in meta.items() if k != "codebook"}
    geometry["rope_theta"] = float(geometry["rope_theta"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"container_bytes": len(raw), "geometry": geometry, "records": records},
        indent=1,
    ) + "\n")
    print(f"wrote {OUT} — {n_rec} records, {len(raw)} container bytes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `PYTHONPATH=needle .venv-parity/bin/python tools/gen_cact_v3_parity.py`
Expected: `wrote crates/needle-infer/tests/cact_v3_vectors.json — 581 records, 35335380 container bytes`

- [ ] **Step 3: Write the failing parity test**

```rust
//! The parsed v3 geometry must match upstream's own reader field for field.
//!
//! The directory is nameless and positional, so a correct tensor in the
//! wrong slot passes every per-tensor check. Agreeing with `read_export`
//! on geometry and on every record's shape is what makes the canon
//! checkable at all.
use std::path::Path;

use needle_infer::cact::{Cact, CactGen};

const VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cact_v3_vectors.json");
const CONTAINER: &str = "weights/needle3.cact";

fn skip(reason: &str) {
    println!("SKIP cact_v3_parity: {reason}");
}

#[test]
fn geometry_matches_upstream_reader() {
    if !Path::new(VECTORS).exists() {
        return skip("no cact_v3_vectors.json; run tools/gen_cact_v3_parity.py");
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let container = root.join(CONTAINER);
    if !container.exists() {
        return skip("no weights/needle3.cact");
    }

    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).unwrap()).unwrap();
    let g = &want["geometry"];

    let cact = Cact::load(&container).expect("v3 container should load");
    let geo = match cact.generation() {
        CactGen::V3(geo) => geo,
        CactGen::V2(_) => panic!("needle3.cact parsed as v2"),
    };

    assert_eq!(geo.num_tensors, g["num_tensors"].as_u64().unwrap() as usize);
    assert_eq!(geo.vocab_size, g["vocab_size"].as_u64().unwrap() as usize);
    assert_eq!(geo.out_vocab, g["out_vocab"].as_u64().unwrap() as usize);
    assert_eq!(geo.d_model, g["d_model"].as_u64().unwrap() as usize);
    assert_eq!(geo.num_heads, g["num_heads"].as_u64().unwrap() as usize);
    assert_eq!(geo.num_kv_heads, g["num_kv_heads"].as_u64().unwrap() as usize);
    assert_eq!(geo.num_layers, g["num_layers"].as_u64().unwrap() as usize);
    assert_eq!(geo.qk_head_dim, g["qk_head_dim"].as_u64().unwrap() as usize);
    assert_eq!(geo.v_head_dim, g["v_head_dim"].as_u64().unwrap() as usize);
    assert_eq!(geo.max_seq_len, g["max_seq_len"].as_u64().unwrap() as usize);
    assert_eq!(geo.hada_n, g["hada_n"].as_u64().unwrap() as usize);
    assert_eq!(geo.mhc_lanes, g["mhc_lanes"].as_u64().unwrap() as usize);
    assert_eq!(geo.sliding_window, g["sliding_window"].as_u64().unwrap() as usize);
    assert_eq!(geo.qkv_conv_taps, g["qkv_conv_taps"].as_u64().unwrap() as usize);
    assert_eq!(geo.engram_slots, g["engram_slots"].as_u64().unwrap() as usize);
    assert_eq!(geo.engram_sub_dim, g["engram_sub_dim"].as_u64().unwrap() as usize);
    assert_eq!(
        geo.num_engram_tables,
        g["num_engram_tables"].as_u64().unwrap() as usize
    );
    assert_eq!(geo.engram_seed_heads, g["engram_seed_heads"].as_u64().unwrap() as usize);
    assert_eq!(geo.kv_window, g["kv_window"].as_u64().unwrap() as usize);
    assert_eq!(geo.kv_bits, g["kv_bits"].as_u64().unwrap() as u32);
    assert_eq!(geo.rope_theta, g["rope_theta"].as_f64().unwrap() as f32);

    let want_u = |k: &str| -> Vec<usize> {
        g[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    };
    assert_eq!(geo.engram_orders, want_u("engram_orders"));
    assert_eq!(geo.engram_sites, want_u("engram_layers"));
    assert_eq!(geo.global_layers(), want_u("global_layers"));

    println!(
        "v3 geometry matches upstream: {} layers x {}, {} tensors",
        geo.num_layers, geo.d_model, geo.num_tensors
    );
}

#[test]
fn every_directory_record_matches() {
    if !Path::new(VECTORS).exists() {
        return skip("no cact_v3_vectors.json");
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let container = root.join(CONTAINER);
    if !container.exists() {
        return skip("no weights/needle3.cact");
    }
    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).unwrap()).unwrap();
    let cact = Cact::load(&container).expect("load");

    let recs = want["records"].as_array().unwrap();
    assert_eq!(cact.records().len(), recs.len(), "record count");
    for (i, w) in recs.iter().enumerate() {
        let r = &cact.records()[i];
        assert_eq!(r.dtype, w["dtype"].as_u64().unwrap() as u8, "dtype at {i}");
        assert_eq!(r.ndim, w["ndim"].as_u64().unwrap() as u8, "ndim at {i}");
        let shape: Vec<usize> = w["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(&r.shape[..r.ndim as usize], &shape[..], "shape at {i}");
    }
    println!("all {} directory records match", recs.len());
}
```

- [ ] **Step 4: Run it and watch it fail**

Run: `cargo test -p needle-infer --test cact_v3_parity -- --nocapture`
Expected: FAIL to compile — `Cact::generation` and `Cact::records` do not exist yet, and `CactGen` is not exported.

- [ ] **Step 5: Wire dispatch into `Cact`**

In `cact.rs`, add the generation enum and accessors, and branch `from_bytes` on the tag. Keep the existing v2 path exactly as it is:

```rust
/// Which generation a container declares.
#[derive(Debug, Clone, PartialEq)]
pub enum CactGen {
    V2(CactGeometry),
    V3(CactV3Geometry),
}

impl Cact {
    /// Geometry, tagged with the generation it came from.
    pub fn generation(&self) -> &CactGen {
        &self.gen
    }

    /// Directory records, in canon order.
    pub fn records(&self) -> &[Record] {
        &self.records
    }
}
```

In `from_bytes`, read the tag first and pick the header width:

```rust
let tag = u(0);
let (gen, header_bytes) = match tag {
    TAG => {
        // ... existing v2 geometry construction, unchanged ...
        (CactGen::V2(geom), HEADER_BYTES)
    }
    TAG_V3 => {
        if raw.len() < HEADER_BYTES_V3 {
            return Err(CactError::TooShort {
                need: HEADER_BYTES_V3,
                got: raw.len(),
            });
        }
        let words: Vec<u32> = (0..49).map(u).collect();
        (CactGen::V3(CactV3Geometry::from_words(&words)?), HEADER_BYTES_V3)
    }
    other => return Err(CactError::BadTag(other)),
};
```

Replace the later uses of the `HEADER_BYTES` constant in directory offset maths with the `header_bytes` local, so the shared directory/codebook/blob code serves both generations.

- [ ] **Step 6: Run the parity tests**

Run: `cargo test -p needle-infer --test cact_v3_parity -- --nocapture`
Expected: PASS, printing `v3 geometry matches upstream: 20 layers x 768, 581 tensors` and `all 581 directory records match`.

- [ ] **Step 7: Confirm v2 still loads**

Run: `cargo test -p needle-infer --test cact_parity -- --nocapture`
Expected: PASS. If this fails, the dispatch refactor broke v2 — stop and fix before continuing.

- [ ] **Step 8: Format, lint, commit**

```bash
cargo fmt --all
cargo clippy -p needle-infer --all-targets -- -D warnings
git add tools/gen_cact_v3_parity.py crates/needle-infer/tests/cact_v3_parity.rs \
        crates/needle-infer/tests/cact_v3_vectors.json crates/needle-infer/src/cact.rs
git commit -m "cact: dispatch on the container tag, verify v3 against upstream

Geometry and all 581 directory records are checked against
export.read_export rather than against documentation. The v2 path is
byte-identical; only the header offset is now a local."
```

---

### Task 3: Confirm the embedded v3 tokenizer round-trips

**Files:**
- Create: `tools/gen_tokenizer_v3_parity.py`
- Create: `crates/needle-infer/tests/tokenizer_v3_parity.rs`

**Interfaces:**
- Consumes: `Cact::records()`, the existing `sp_tokenizer::SpTokenizer`.
- Produces: `crates/needle-infer/tests/tokenizer_v3_vectors.json` — `{"cases": [{"text": str, "ids": [u32]}]}`.

Upstream's `_TK_HDR` is unchanged between v2 and v3, so the existing tokenizer should work as-is. This task **verifies** that rather than assuming it; if it fails, the failure is the finding.

- [ ] **Step 1: Write the generator**

```python
#!/usr/bin/env python3
"""Emit v3 tokenizer parity vectors from the container's embedded blob.

    PYTHONPATH=needle .venv-parity/bin/python tools/gen_tokenizer_v3_parity.py
"""
import json
import pathlib
import sys

from needle.model import export

CONTAINER = pathlib.Path("weights/needle3.cact")
OUT = pathlib.Path("crates/needle-infer/tests/tokenizer_v3_vectors.json")

CASES = [
    "What's the weather in Paris?",
    "Turn off the bedroom lights",
    "Book a flight from London to JFK tomorrow",
    "Email alice@example.com saying the build is green",
    "<|im_start|>user\n<tools>[]</tools>\nhi<|im_end|>\n<|im_start|>assistant\n",
    "set(x) := {1, 2, 3}",
    "   leading and trailing   ",
    "naïve café — résumé",
    "日本語のテキスト",
    "",
]


def main():
    if not CONTAINER.exists():
        sys.exit(f"missing {CONTAINER}")
    blob = export.read_tokenizer_blob(str(CONTAINER))
    tok = export.parse_tokenizer_blob(blob)
    print(f"pieces={tok['pieces']} pad={tok['pad_id']}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load("weights/tokenizer.model")

    cases = [{"text": t, "ids": [int(i) for i in sp.encode(t)]} for t in CASES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pieces": tok["pieces"], "cases": cases}, indent=1) + "\n")
    print(f"wrote {OUT} — {len(cases)} cases")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fetch the v3 tokenizer model and run the generator**

```bash
hf download Cactus-Compute/needle3 tokenizer/tokenizer.model --local-dir weights/
PYTHONPATH=needle .venv-parity/bin/python tools/gen_tokenizer_v3_parity.py
```

Expected: prints the piece count and `wrote crates/needle-infer/tests/tokenizer_v3_vectors.json — 10 cases`.

- [ ] **Step 3: Write the failing test**

```rust
//! The v3 container embeds its own SentencePiece model. Upstream's
//! tokenizer header format is unchanged from v2, so the existing reader
//! should serve it — this test is what makes that a fact rather than an
//! assumption.
use std::path::Path;

use needle_infer::cact::Cact;
use needle_infer::sp_tokenizer::SpTokenizer;

const VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tokenizer_v3_vectors.json");

#[test]
fn v3_embedded_tokenizer_matches_sentencepiece() {
    if !Path::new(VECTORS).exists() {
        println!("SKIP: no tokenizer_v3_vectors.json; run tools/gen_tokenizer_v3_parity.py");
        return;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root");
    let container = root.join("weights/needle3.cact");
    if !container.exists() {
        println!("SKIP: no weights/needle3.cact");
        return;
    }

    let cact = Cact::load(&container).expect("load");
    let blob = cact.tokenizer_blob().expect("v3 container carries a tokenizer blob");
    let tok = SpTokenizer::from_blob(blob).expect("tokenizer blob should parse");

    let want: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).unwrap()).unwrap();
    assert_eq!(
        tok.len(),
        want["pieces"].as_u64().unwrap() as usize,
        "piece count"
    );

    let mut checked = 0usize;
    for case in want["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expect: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tok.encode(text), expect, "token ids for {text:?}");
        checked += 1;
    }
    println!("v3 tokenizer: {checked} cases match sentencepiece exactly");
}
```

- [ ] **Step 4: Run it**

Run: `cargo test -p needle-infer --test tokenizer_v3_parity -- --nocapture`
Expected: either PASS (the v2 reader serves v3 unchanged — record that in the port record), or a concrete mismatch. If it fails, do not adapt the test: read `export.parse_tokenizer_blob` against `sp_tokenizer.rs` and fix the reader, because a tokenizer difference would otherwise surface much later as wrong logits.

Note: if `Cact::tokenizer_blob()` does not exist yet, add it — it returns the single `RAW` record's bytes, of which the v3 container has exactly one.

- [ ] **Step 5: Format, lint, commit**

```bash
cargo fmt --all
cargo clippy -p needle-infer --all-targets -- -D warnings
git add tools/gen_tokenizer_v3_parity.py \
        crates/needle-infer/tests/tokenizer_v3_parity.rs \
        crates/needle-infer/tests/tokenizer_v3_vectors.json
git commit -m "tokenizer: verify the v3 embedded blob against sentencepiece

Upstream's tokenizer header is unchanged from v2, so the existing reader
serves v3. Asserted on a 10-case corpus rather than assumed."
```

---

### Task 4: Wire the v3 suites into CI

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the test binaries from Tasks 2 and 3.
- Produces: a `parity-v3` job, on the `parity-v2` pattern.

- [ ] **Step 1: Add the job**

Append to `.github/workflows/ci.yml`, mirroring the existing `parity-v2` job:

```yaml
  parity-v3:
    name: v3 parity tests (needle3.cact)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # Public repo, no auth needed. 35.3 MB.
      - name: Download the v3 checkpoint from HuggingFace
        run: |
          pip install -q huggingface_hub
          hf download Cactus-Compute/needle3 needle3.cact --local-dir weights/

      # Container and tokenizer fixtures are tracked, so these need no JAX.
      # Runs on x86_64, so it also exercises the AVX2 dispatch in
      # needle-core::cq against the scalar reference.
      - name: Run v3 parity tests
        run: |
          cargo test -p needle-infer --release --test cact_v3_parity -- --nocapture
          cargo test -p needle-infer --release --test tokenizer_v3_parity -- --nocapture
```

- [ ] **Step 2: Validate the workflow parses**

Run: `python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml')); print(len(d['jobs']), 'jobs:', sorted(d['jobs']))"`
Expected: the job count increases by one and `parity-v3` is listed.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: gate the v3 container and tokenizer against the real checkpoint"
```

---

## Self-Review

**Spec coverage.** This plan implements spec milestones 1 (container) and 2 (tokenizer), plus the `parity-v3` CI job from milestone 8. Milestones 3–7 (forward pass, engine/CLI, performance, heads and constrained decoding, bindings) and the rest of milestone 8 (docs, sizes, the banner, licence wording) are **not** covered here and need their own plans. Milestone 3's plan is blocked on spec Risk 1 — how global-attention layers allocate KV when the ring is sized to `kv_window` — which must be decided before it can be written without placeholders.

**Placeholder scan.** No TBD/TODO. Every code step carries real code. The one conditional instruction (Task 3 Step 4, "if it fails, fix the reader") is a genuine branch on an empirical result, not a deferred decision — and it names exactly which two files to compare.

**Type consistency.** `CactV3Geometry::from_words(&[u32])` is defined in Task 1 and consumed in Task 2. `CactGen::{V2,V3}`, `Cact::generation()` and `Cact::records()` are introduced in Task 2 Step 5 and used by the tests written in Task 2 Step 3 — the test is written first deliberately, so the compile failure in Step 4 is the expected TDD red. `Cact::tokenizer_blob()` is flagged in Task 3 Step 4 as possibly needing to be added. `engram_sites` in Rust maps to upstream's `engram_layers` key, which the Task 2 test handles explicitly.
