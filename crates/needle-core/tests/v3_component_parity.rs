//! Needle 3 component parity, against upstream's own Flax modules.
//!
//! Needs `tests/v3_component_vectors.json` + `.f32` from
//! `tools/gen_v3_component_parity.py`. Each component is checked on its own,
//! so a sign or transpose error reads as "HadamardMLP is wrong" rather than
//! surfacing twenty layers later as wrong logits.

use std::path::Path;

use needle_core::v3::kernels::{hada_blocks, hadamard_mlp, HadaMlp, HadaPerms};

const JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_component_vectors.json"
);
const F32: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/v3_component_vectors.f32"
);

struct Fixture {
    meta: serde_json::Value,
    floats: Vec<f32>,
}

impl Fixture {
    fn load() -> Option<Self> {
        if !Path::new(JSON).exists() || !Path::new(F32).exists() {
            println!(
                "skipping v3 component parity: run\n  \
                 JAX_PLATFORMS=cpu PYTHONPATH=needle:tools .venv-parity/bin/python \
                 tools/gen_v3_component_parity.py"
            );
            return None;
        }
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(JSON).unwrap()).unwrap();
        let raw = std::fs::read(F32).unwrap();
        let floats = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(Self { meta, floats })
    }

    fn get(&self, name: &str) -> Vec<f32> {
        let e = &self.meta["components"][name];
        let off = e["offset"].as_u64().unwrap() as usize;
        let len = e["len"].as_u64().unwrap() as usize;
        self.floats[off..off + len].to_vec()
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.meta["components"][name]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }
}

#[test]
fn hadamard_mlp_matches_the_reference() {
    let Some(f) = Fixture::load() else { return };

    let d_model = f.meta["geometry"]["d_model"].as_u64().unwrap() as usize;
    let hada_n = f.meta["geometry"]["hada_n"].as_u64().unwrap() as usize;
    let cond_rank = f.meta["geometry"]["cond_rank"].as_u64().unwrap() as usize;
    assert_eq!(hada_blocks(hada_n), (32, 32), "block sizes");

    let perms = HadaPerms {
        p1: f.get("hada_p1").iter().map(|&v| v as u32).collect(),
        p2: f.get("hada_p2").iter().map(|&v| v as u32).collect(),
    };
    let w = HadaMlp {
        d1: f.get("mlp_d1"),
        d2: f.get("mlp_d2"),
        b2: f.get("mlp_b2"),
        d3: f.get("mlp_d3"),
        d4: f.get("mlp_d4"),
        w1: (f.get("mlp_w1a"), f.get("mlp_w1b")),
        w2: (f.get("mlp_w2a"), f.get("mlp_w2b")),
        w3: (f.get("mlp_w3a"), f.get("mlp_w3b")),
        cond_v: f.get("mlp_cond_v"),
        cond_u: f.get("mlp_cond_u"),
        cond_rank,
    };

    let x_all = f.get("mlp_in");
    let want_all = f.get("mlp_out");
    let seq = f.shape("mlp_in")[0];
    assert_eq!(x_all.len(), seq * d_model);

    // Deviation is measured against the output RMS, not per element. A
    // per-element relative metric explodes on near-zero outputs and says
    // nothing useful; the fixture is generated in float32 precisely so this
    // number reflects the arithmetic rather than the reference's dtype.
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    let mut sq = 0.0f64;
    for t in 0..seq {
        let x = &x_all[t * d_model..(t + 1) * d_model];
        let want = &want_all[t * d_model..(t + 1) * d_model];
        let mut got = vec![0.0f32; d_model];
        hadamard_mlp(x, &w, &perms, hada_n, &mut got);

        for (i, (&g, &e)) in got.iter().zip(want).enumerate() {
            sq += (e as f64) * (e as f64);
            let d = (g - e).abs();
            if d > worst {
                worst = d;
                worst_at = (t, i);
            }
        }
    }
    let rms = (sq / (seq * d_model) as f64).sqrt() as f32;
    let rel = worst / rms;

    println!(
        "HadamardMLP: max abs deviation {worst:.3e} against output RMS {rms:.1} \
         = {rel:.3e} relative (worst at position {} dim {})",
        worst_at.0, worst_at.1
    );
    assert!(
        rel < 1e-4,
        "HadamardMLP deviates from the f32 reference by {rel:.3e} of RMS \
         (abs {worst:.3e}) at position {} dim {}",
        worst_at.0,
        worst_at.1
    );
}
