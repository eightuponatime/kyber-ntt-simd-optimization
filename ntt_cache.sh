#!/bin/bash

# turn off NMI watchdog
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null

# binary path finder
BENCH_BIN=$(find target/release/deps -name "ntt_bench-*" -type f -executable | head -n 1)

if [ -z "$BENCH_BIN" ]; then
    echo "Error: Benchmark binary not found. Run 'cargo bench --no-run' first."
    exit 1
fi

echo "Using benchmark binary: $BENCH_BIN"
echo ""

# === BASIC NTT ===
echo "=== BASIC NTT ==="
perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_basic_ntt_forward

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_basic_ntt_inverse

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_basic_ntt_roundtrip

# === SIMD NTT ===
echo ""
echo "=== SIMD NTT ==="
perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_ntt_forward

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_ntt_inverse

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_ntt_roundtrip

# === SIMD NTT OPTIMIZED ===
echo ""
echo "=== SIMD NTT OPTIMIZED ==="
perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_ntt_opt_forward

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_ntt_opt_inverse

perf stat -e cache-misses,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
    "$BENCH_BIN" --bench bench_simd_ntt_opt_roundtrip

# turn on NMI watchdog
echo 1 | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null

echo ""
echo "=== DONE ==="
