use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ntt_simd_optimization::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};

#[cfg(target_arch = "x86_64")]
use ntt_simd_optimization::simd_ops::{
    simd_barrett_reduce_8, simd_mod_add_8, simd_mod_mul_8, simd_mod_sub_8,
};

#[cfg(target_arch = "x86_64")]
use ntt_simd_optimization::simd_ops_opt::{
    simd_barrett_reduce_8 as simd_barrett_reduce_8_opt, simd_barrett_reduce_8_register_opt,
    simd_mod_add_8 as simd_mod_add_8_opt, simd_mod_mul_8 as simd_mod_mul_8_opt,
    simd_mod_sub_8 as simd_mod_sub_8_opt,
};

// Scalar baseline
fn bench_scalar_add(c: &mut Criterion) {
    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("scalar_add", |bencher| {
        bencher.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = mod_add(black_box(a[i]), black_box(b[i]));
            }
            black_box(result);
        });
    });
}

fn bench_scalar_sub(c: &mut Criterion) {
    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("scalar_sub", |bencher| {
        bencher.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = mod_sub(black_box(a[i]), black_box(b[i]));
            }
            black_box(result);
        });
    });
}

fn bench_scalar_mul(c: &mut Criterion) {
    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("scalar_mul", |bencher| {
        bencher.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = mod_mul(black_box(a[i]), black_box(b[i]));
            }
            black_box(result);
        });
    });
}

fn bench_scalar_barrett(c: &mut Criterion) {
    let a = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];

    c.bench_function("scalar_barrett", |bencher| {
        bencher.iter(|| {
            let mut result = [0i32; 8];
            for i in 0..8 {
                result[i] = barrett_reduce(black_box(a[i]));
            }
            black_box(result);
        });
    });
}

// SIMD basic
#[cfg(target_arch = "x86_64")]
fn bench_simd_add(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("simd_add", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_mod_add_8(black_box(&a), black_box(&b)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_sub(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("simd_sub", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_mod_sub_8(black_box(&a), black_box(&b)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_mul(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("simd_mul", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_mod_mul_8(black_box(&a), black_box(&b)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_barrett(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];

    c.bench_function("simd_barrett", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_barrett_reduce_8(black_box(&a)));
        });
    });
}

// SIMD optimized
#[cfg(target_arch = "x86_64")]
fn bench_simd_add_opt(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("simd_add_opt", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_mod_add_8_opt(black_box(&a), black_box(&b)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_sub_opt(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("simd_sub_opt", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_mod_sub_8_opt(black_box(&a), black_box(&b)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_mul_opt(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [10, 20, 30, 40, 50, 60, 70, 80];
    let b = [5, 10, 15, 20, 25, 30, 35, 40];

    c.bench_function("simd_mul_opt", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_mod_mul_8_opt(black_box(&a), black_box(&b)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_barrett_opt(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];

    c.bench_function("simd_barrett_opt", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_barrett_reduce_8_opt(black_box(&a)));
        });
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_simd_barrett_register_opt(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let a = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];

    c.bench_function("simd_barrett_register_opt", |bencher| {
        bencher.iter(|| unsafe {
            black_box(simd_barrett_reduce_8_register_opt(black_box(&a)));
        });
    });
}

criterion_group!(
    simd_ops,
    bench_scalar_add,
    bench_scalar_sub,
    bench_scalar_mul,
    bench_scalar_barrett,
);

#[cfg(target_arch = "x86_64")]
criterion_group!(
    simd_ops_avx2,
    bench_simd_add,
    bench_simd_sub,
    bench_simd_mul,
    bench_simd_barrett,
    bench_simd_add_opt,
    bench_simd_sub_opt,
    bench_simd_mul_opt,
    bench_simd_barrett_opt,
    bench_simd_barrett_register_opt,
);

#[cfg(target_arch = "x86_64")]
criterion_main!(simd_ops, simd_ops_avx2);

#[cfg(not(target_arch = "x86_64"))]
criterion_main!(simd_ops);
