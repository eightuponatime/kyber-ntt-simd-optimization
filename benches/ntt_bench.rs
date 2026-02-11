use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ntt_simd_optimization::{BasicNTT, SimdNTT, SimdNttOpt, NTT};

fn bench_basic_ntt_forward(c: &mut Criterion) {
    let ntt = BasicNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    c.bench_function("basic_ntt_forward", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.forward(black_box(&mut p));
            black_box(p);
        });
    });
}

fn bench_basic_ntt_inverse(c: &mut Criterion) {
    let ntt = BasicNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    // Transform to frequency domain first
    ntt.forward(&mut poly);

    c.bench_function("basic_ntt_inverse", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.inverse(black_box(&mut p));
            black_box(p);
        });
    });
}

fn bench_basic_ntt_roundtrip(c: &mut Criterion) {
    let ntt = BasicNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    c.bench_function("basic_ntt_roundtrip", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.forward(black_box(&mut p));
            ntt.inverse(black_box(&mut p));
            black_box(p);
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_ntt_forward(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let ntt = SimdNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    c.bench_function("simd_ntt_forward", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.forward(black_box(&mut p));
            black_box(p);
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_ntt_inverse(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let ntt = SimdNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    ntt.forward(&mut poly);

    c.bench_function("simd_ntt_inverse", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.inverse(black_box(&mut p));
            black_box(p);
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_ntt_roundtrip(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let ntt = SimdNTT::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    c.bench_function("simd_ntt_roundtrip", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.forward(black_box(&mut p));
            ntt.inverse(black_box(&mut p));
            black_box(p);
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_ntt_opt_forward(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let ntt = SimdNttOpt::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    c.bench_function("simd_ntt_opt_forward", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.forward(black_box(&mut p));
            black_box(p);
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_ntt_opt_inverse(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let ntt = SimdNttOpt::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    ntt.forward(&mut poly);

    c.bench_function("simd_ntt_opt_inverse", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.inverse(black_box(&mut p));
            black_box(p);
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_ntt_opt_roundtrip(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let ntt = SimdNttOpt::new();
    let mut poly = [0i32; 256];

    for i in 0..256 {
        poly[i] = (i as i32 * 13) % 3329;
    }

    c.bench_function("simd_ntt_opt_roundtrip", |bencher| {
        bencher.iter(|| {
            let mut p = poly;
            ntt.forward(black_box(&mut p));
            ntt.inverse(black_box(&mut p));
            black_box(p);
        });
    });
}

criterion_group!(
    ntt_basic,
    bench_basic_ntt_forward,
    bench_basic_ntt_inverse,
    bench_basic_ntt_roundtrip,
);

#[cfg(target_arch = "x86_64")]
criterion_group!(
    ntt_simd,
    bench_simd_ntt_forward,
    bench_simd_ntt_inverse,
    bench_simd_ntt_roundtrip,
    bench_simd_ntt_opt_forward,
    bench_simd_ntt_opt_inverse,
    bench_simd_ntt_opt_roundtrip,
);

#[cfg(target_arch = "x86_64")]
criterion_main!(ntt_basic, ntt_simd);

#[cfg(not(target_arch = "x86_64"))]
criterion_main!(ntt_basic);
