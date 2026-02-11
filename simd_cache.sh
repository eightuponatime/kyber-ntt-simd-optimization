#!/bin/bash

# turning off NMI watchdog for accurate benches
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null

# binary path finder
BENCH_BIN=$(find target/release/deps -name "simd_ops_bench-*" -type f -executable | head -n 1)

if [ -z "$BENCH_BIN" ]; then
    echo "Error: Benchmark binary not found. Run 'cargo bench --no-run' first."
    exit 1
fi

echo "Using benchmark binary: $BENCH_BIN"
echo ""

# === SCALAR ===
echo "=== SCALAR OPERATIONS ==="
perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_scalar_add

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_scalar_sub

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_scalar_mul

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_scalar_barrett

# === SIMD BASIC ===
echo ""
echo "=== SIMD BASIC OPERATIONS ==="
perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_add

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_sub

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_mul

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_barrett

# === SIMD OPTIMIZED ===
echo ""
echo "=== SIMD OPTIMIZED OPERATIONS ==="
perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_add_opt

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_sub_opt

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_mul_opt

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_barrett_opt

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_barrett_register_opt

# turn on NMI watchdog
echo 1 | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null

echo ""
echo "=== DONE ==="
