//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench llvm_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ntt_simd_optimization::llvm_demo;

fn gen_data(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i as i32 * 7 - 200) % 500 - 100).collect()
}

fn bench_clamp(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096];
    let mut group = c.benchmark_group("clamp");

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, &n| {
            let mut data = gen_data(n);
            b.iter(|| {
                // восстанавливаем данные каждую итерацию
                for (i, x) in data.iter_mut().enumerate() {
                    *x = (i as i32 * 7 - 200) % 500 - 100;
                }
                llvm_demo::clamp_scalar(black_box(&mut data));
            });
        });

        group.bench_with_input(BenchmarkId::new("inline_asm", size), &size, |b, &n| {
            let mut data = gen_data(n);
            b.iter(|| {
                for (i, x) in data.iter_mut().enumerate() {
                    *x = (i as i32 * 7 - 200) % 500 - 100;
                }
                unsafe { llvm_demo::clamp_inline_asm(black_box(&mut data)); }
            });
        });

        group.bench_with_input(BenchmarkId::new("intrinsics", size), &size, |b, &n| {
            let mut data = gen_data(n);
            b.iter(|| {
                for (i, x) in data.iter_mut().enumerate() {
                    *x = (i as i32 * 7 - 200) % 500 - 100;
                }
                unsafe { llvm_demo::clamp_intrinsics(black_box(&mut data)); }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_clamp);
criterion_main!(benches);