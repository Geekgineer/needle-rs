use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use needle_core::norm::zc_rms_norm_vec;
use needle_core::quant::QuantizedWeight;
use needle_core::rope::RopeCache;

// ── QuantizedWeight::matvec ───────────────────────────────────────────────────

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_int4");

    // Production-size projections: (in, out)
    let shapes: &[(usize, usize, &str)] = &[
        (512, 512, "512x512"),   // Q/K/V proj, d=512
        (512, 256, "512x256"),   // K/V with num_kv_heads=4, head_dim=64
        (2048, 512, "2048x512"), // FFN down-proj (d_ff=2048, d=512)
        (512, 2048, "512x2048"), // FFN up-proj
        (16, 16, "16x16"),       // tiny test config baseline
    ];

    for &(in_feat, out_feat, label) in shapes {
        let w: Vec<f32> = (0..in_feat * out_feat)
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let x: Vec<f32> = (0..in_feat).map(|i| (i as f32).cos()).collect();
        let qw = QuantizedWeight::quantize(&w, in_feat, out_feat);
        let mut y = vec![0.0f32; out_feat];

        group.throughput(Throughput::Elements((in_feat * out_feat) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), label, |b, _| {
            b.iter(|| {
                qw.matvec(black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

// ── ZCRMSNorm ────────────────────────────────────────────────────────────────

fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("zc_rms_norm");

    for d in [16usize, 512, 2048] {
        let scale = vec![0.0f32; d];
        let mut x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01 + 1.0).collect();

        group.throughput(Throughput::Elements(d as u64));
        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |b, _| {
            b.iter(|| {
                zc_rms_norm_vec(black_box(&mut x), black_box(&scale));
            });
        });
    }

    group.finish();
}

// ── RoPE::apply ──────────────────────────────────────────────────────────────

fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_apply");

    // Typical production: 8 heads, head_dim=64, seq_len=128
    let (num_heads, head_dim, seq_len) = (8, 64, 128);
    let rope = RopeCache::new(1024, head_dim, 10000.0);
    let mut q = vec![1.0f32; num_heads * head_dim];

    group.throughput(Throughput::Elements((num_heads * head_dim) as u64));
    group.bench_function("8h_hd64_pos0", |b| {
        b.iter(|| {
            rope.apply(black_box(&mut q), black_box(num_heads), 1, head_dim, 0);
        });
    });

    // Batch over seq_len (simulate full encoder pass)
    let mut q_seq = vec![1.0f32; seq_len * num_heads * head_dim];
    group.bench_function("8h_hd64_seq128", |b| {
        b.iter(|| {
            for pos in 0..seq_len {
                let start = pos * num_heads * head_dim;
                let end = start + num_heads * head_dim;
                rope.apply(
                    black_box(&mut q_seq[start..end]),
                    num_heads,
                    1,
                    head_dim,
                    pos,
                );
            }
        });
    });

    group.finish();
}

// ── QuantizedWeight::matmul (batched, for encoder) ───────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_int4_batched");

    // Encoder full-sequence projection: batch=128 tokens, 512→512
    let (in_feat, out_feat) = (512, 512);
    let w: Vec<f32> = (0..in_feat * out_feat)
        .map(|i| (i as f32).sin() * 0.1)
        .collect();
    let qw = QuantizedWeight::quantize(&w, in_feat, out_feat);

    for batch in [1usize, 16, 64, 128] {
        let x = vec![0.5f32; batch * in_feat];
        let mut y = vec![0.0f32; batch * out_feat];

        group.throughput(Throughput::Elements((batch * in_feat * out_feat) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, _| {
            b.iter(|| {
                qw.matmul(black_box(&x), batch, black_box(&mut y));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matvec,
    bench_rms_norm,
    bench_rope,
    bench_matmul
);
criterion_main!(benches);
