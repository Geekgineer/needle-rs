//! The int8 KV quantiser, against upstream's `fake_quant`.
//!
//! Needs `tests/v3_kv_quant_vectors.json`, generated from
//! `needle.model.quantize.fake_quant(x, x.shape[-1], 8)` — which is exactly
//! what `a8_fake_quant_kv` calls.
//!
//! The cases include the ones that break a naive implementation: an all-zero
//! vector (scale must stay 1, not divide by zero), a single spike (every other
//! element quantises to zero), values at 1e-8 and at 1e6.

use needle_core::v3::fake_quant_vec;

const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_kv_quant_vectors.json"
);

#[test]
fn matches_upstream_fake_quant() {
    if !std::path::Path::new(VECTORS).exists() {
        println!("skipping: no tests/v3_kv_quant_vectors.json");
        return;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).unwrap()).unwrap();
    let bits = v["bits"].as_u64().unwrap() as u32;

    for case in v["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let dim = case["dim"].as_u64().unwrap() as usize;
        let read = |k: &str| -> Vec<f32> {
            case[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let input = read("in");
        let want = read("out");
        assert_eq!(input.len() % dim, 0);

        let mut got = input.clone();
        for chunk in got.chunks_mut(dim) {
            fake_quant_vec(chunk, bits);
        }

        let mut worst = 0.0f32;
        for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
            }
            // Exactness matters here: the quantiser is a rounding rule, so a
            // mismatch is a different rule rather than accumulated error.
            let tol = b.abs().max(1.0) * 1e-5;
            assert!(
                d <= tol,
                "{name}[{i}]: got {a}, want {b} (input {})",
                input[i]
            );
        }
        println!("  {name:16} matches, max abs diff {worst:.3e}");
    }
}

#[test]
fn an_all_zero_vector_survives() {
    // scale would be 0/qmax; upstream guards with absmax > 0 and keeps 1.
    let mut x = vec![0.0f32; 48];
    fake_quant_vec(&mut x, 8);
    assert!(x.iter().all(|v| *v == 0.0), "zeros must stay zero");
}

#[test]
fn the_error_is_bounded_by_the_step() {
    // Symmetric int8 over a vector with absmax A has step A/127, so no element
    // may move more than half a step.
    let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
    let absmax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut q = x.clone();
    fake_quant_vec(&mut q, 8);
    let step = absmax / 127.0;
    for (i, (&a, &b)) in q.iter().zip(&x).enumerate() {
        assert!(
            (a - b).abs() <= step * 0.5 + 1e-6,
            "element {i} moved {}, more than half a step {}",
            (a - b).abs(),
            step * 0.5
        );
    }
}

/// The stored-integer reader must agree with the round-trip the fixtures pin.
///
/// `fake_quant_vec` is verified against upstream above, but attention does not
/// call it: it reads stored `i8`s and hoists the per-head scale out of the dot
/// product, computing `scale · Σ(q·i8)` where the reference computes
/// `Σ(q · (i8·scale))`. Those are the same number in algebra and not
/// necessarily the same `f32`. Nothing else covers that step — the quantiser
/// could stay exact while the reader drifted, and every other test would pass.
#[test]
fn the_hoisted_scale_reader_matches_a_dequantised_buffer() {
    use needle_core::v3::{attend_step, fake_quant_vec, AttnDims, KvStore, Ring};

    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const QK: usize = 48;
    const V: usize = 64;
    const SLOTS: usize = 17;

    let d = AttnDims {
        seq: 1,
        num_heads: HEADS,
        num_kv_heads: KV_HEADS,
        qk_head_dim: QK,
        v_head_dim: V,
    };

    // Deterministic, and deliberately spanning several magnitudes per head so
    // each head gets a different scale.
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };

    let q: Vec<f32> = (0..HEADS * QK).map(|_| next()).collect();
    // Alternate the magnitude head by head so neighbouring heads get scales
    // orders of magnitude apart — a single shared scale would pass otherwise.
    let spread = |rows: usize, width: usize, lo: f32, hi: f32, next: &mut dyn FnMut() -> f32| {
        let mut out = Vec::with_capacity(rows * width);
        for r in 0..rows {
            let m = if r.is_multiple_of(2) { lo } else { hi };
            for _ in 0..width {
                out.push(next() * m);
            }
        }
        out
    };
    let k_raw = spread(SLOTS * KV_HEADS, QK, 1.0, 40.0, &mut next);
    let v_raw = spread(SLOTS * KV_HEADS, V, 0.01, 3.0, &mut next);

    // Reference: dequantised f32, exactly what `fake_quant_vec` leaves behind.
    let mut k_deq = k_raw.clone();
    let mut v_deq = v_raw.clone();
    for c in k_deq.chunks_mut(QK) {
        fake_quant_vec(c, 8);
    }
    for c in v_deq.chunks_mut(V) {
        fake_quant_vec(c, 8);
    }

    // The same values as stored integers plus per-head scales.
    let quant = |src: &[f32], width: usize| -> (Vec<i8>, Vec<f32>) {
        let mut ints = Vec::with_capacity(src.len());
        let mut scales = Vec::with_capacity(src.len() / width);
        for c in src.chunks(width) {
            let absmax = c.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            for x in c {
                ints.push((x / scale).round().clamp(-128.0, 127.0) as i8);
            }
            scales.push(scale);
        }
        (ints, scales)
    };
    let (kq, ks) = quant(&k_raw, QK);
    let (vq, vs) = quant(&v_raw, V);

    for (lo, hi) in [(0usize, 0usize), (0, 9), (3, 16), (5, 5)] {
        let mut a = vec![0.0f32; HEADS * V];
        let mut b = vec![0.0f32; HEADS * V];
        attend_step(
            &q,
            Ring {
                kv: KvStore::F32 {
                    k: &k_deq,
                    v: &v_deq,
                },
                slots: SLOTS,
                lo,
                hi,
            },
            d,
            &mut a,
        );
        attend_step(
            &q,
            Ring {
                kv: KvStore::Int8 {
                    k: &kq,
                    k_scale: &ks,
                    v: &vq,
                    v_scale: &vs,
                },
                slots: SLOTS,
                lo,
                hi,
            },
            d,
            &mut b,
        );
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let rms = (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt();
        println!("  range {lo}..={hi}: max abs diff {worst:.3e} against RMS {rms:.3e}");
        assert!(
            worst < 1e-5 * rms.max(1.0),
            "hoisted-scale reader drifted from the dequantised reference: {worst:.3e}"
        );
    }
}
