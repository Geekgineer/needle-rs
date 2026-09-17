//! Where does a Needle 3 forward pass actually spend its time?
//!
//! Not a benchmark — a probe, so optimisation effort goes where the time is
//! rather than where it is assumed to be. Run with:
//!
//! ```text
//! cargo test -p needle-infer --release --test v3_perf_probe -- --nocapture
//! ```

use std::time::Instant;

use needle_core::v3::kernels::hadamard_mlp;
use needle_infer::cact::CactV3;
use needle_infer::v3::model_from_cact;

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle3.cact");

#[test]
fn where_the_time_goes() {
    if !std::path::Path::new(CACT).exists() {
        println!("skipping perf probe: no weights/needle3.cact");
        return;
    }
    let cact = CactV3::load(CACT).expect("load");
    let model = model_from_cact(&cact).expect("build");
    let cfg = &model.cfg;
    let d = cfg.d_model;
    let seq = 57usize;
    let layer = &model.layers[0];

    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.01).sin()).collect();

    // ── One projection, per position vs batched ──────────────────────────
    let rows = cfg.q_dim();
    let mut y = vec![0.0f32; seq * rows];

    let t = Instant::now();
    for i in 0..seq {
        layer.q_proj.matvec(&x, &mut y[i * rows..(i + 1) * rows]);
    }
    let per_pos = t.elapsed();

    let prep = layer.q_proj.prepared_len();
    let mut xh = vec![0.0f32; seq * prep];
    for i in 0..seq {
        layer
            .q_proj
            .prepare_input(&x, &mut xh[i * prep..(i + 1) * prep]);
    }
    let mut acc = vec![0.0f32; seq];
    let t = Instant::now();
    layer
        .q_proj
        .matmul_rows_prepared(&xh, seq, 0, rows, &mut y, &mut acc);
    let batched = t.elapsed();

    println!(
        "q_proj {rows}x{d} over {seq} positions:\n  \
         per-position matvec {per_pos:?}\n  \
         batched matmul      {batched:?}  ({:.2}x)",
        per_pos.as_secs_f64() / batched.as_secs_f64()
    );

    // ── The five projections one layer runs, per position ────────────────
    let mut scratch = vec![0.0f32; cfg.attn_out_dim().max(d)];
    let t = Instant::now();
    for _ in 0..seq {
        layer.q_proj.matvec(&x, &mut scratch[..cfg.q_dim()]);
        layer.k_proj.matvec(&x, &mut scratch[..cfg.k_dim()]);
        layer.v_proj.matvec(&x, &mut scratch[..cfg.v_dim()]);
        layer
            .gate_proj
            .matvec(&x, &mut scratch[..cfg.attn_out_dim()]);
    }
    let projections = t.elapsed();

    // ── HadamardMLP ──────────────────────────────────────────────────────
    let mut out = vec![0.0f32; d];
    let t = Instant::now();
    for _ in 0..seq {
        hadamard_mlp(&x, &layer.mlp, &model.perms, cfg.hada_n, &mut out);
    }
    let mlp = t.elapsed();

    // ── The tied LM head ─────────────────────────────────────────────────
    let mut logits = vec![0.0f32; cfg.vocab_size];
    let mut lm_prep = vec![0.0f32; model.embedding.prepared_len()];
    let t = Instant::now();
    for _ in 0..seq {
        model.embedding.prepare_input(&x, &mut lm_prep);
        model.embedding.matvec_prepared(&lm_prep, &mut logits);
    }
    let head = t.elapsed();

    let layers = cfg.num_layers as u32;
    let proj_all = projections * layers;
    let mlp_all = mlp * layers;
    let total = proj_all + mlp_all + head;

    println!(
        "\nper forward of {seq} positions (extrapolated from layer 0):\n  \
         projections x{layers:<3} {proj_all:>10.3?}  {:>5.1}%\n  \
         HadamardMLP x{layers:<3} {mlp_all:>10.3?}  {:>5.1}%\n  \
         LM head          {head:>10.3?}  {:>5.1}%\n  \
         accounted        {total:>10.3?}",
        100.0 * proj_all.as_secs_f64() / total.as_secs_f64(),
        100.0 * mlp_all.as_secs_f64() / total.as_secs_f64(),
        100.0 * head.as_secs_f64() / total.as_secs_f64(),
    );
}
