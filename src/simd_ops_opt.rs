//! Register-optimized SIMD operations for modular arithmetic

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::params::Q;

/// SIMD modular addition: (a + b) mod Q for 8 elements
/// Same as v1 - already optimal
#[target_feature(enable = "avx2")]
pub unsafe fn simd_mod_add_8(a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let vsum = _mm256_add_epi32(va, vb);

    let vq = _mm256_set1_epi32(Q);
    let mask = _mm256_cmpgt_epi32(vsum, _mm256_sub_epi32(vq, _mm256_set1_epi32(1)));
    let vq_masked = _mm256_and_si256(mask, vq);
    let result = _mm256_sub_epi32(vsum, vq_masked);

    let mut output = [0i32; 8];
    _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
    output
}

/// same as v1 already optimal
#[target_feature(enable = "avx2")]
pub unsafe fn simd_mod_sub_8(a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let vdiff = _mm256_sub_epi32(va, vb);

    let vq = _mm256_set1_epi32(Q);
    let vzero = _mm256_setzero_si256();
    let mask = _mm256_cmpgt_epi32(vzero, vdiff);
    let vq_masked = _mm256_and_si256(mask, vq);
    let result = _mm256_add_epi32(vdiff, vq_masked);

    let mut output = [0i32; 8];
    _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
    output
}

/// Register-optimized SIMD modular multiplication
///
/// Key optimization: Use shuffle/blend instructions to pack results
/// instead of store > scalar extract > pack pattern
#[target_feature(enable = "avx2")]
pub unsafe fn simd_mod_mul_8(a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let vqinv = _mm256_set1_epi64x(-3327);
    let vq64 = _mm256_set1_epi64x(Q as i64);
    let mask_16bit = _mm256_set1_epi64x(0xFFFF);

    // === Even elements [0, 2, 4, 6] ===
    let prod64_even = _mm256_mul_epi32(va, vb);
    let t_even = _mm256_mul_epi32(prod64_even, vqinv);
    let t_even_masked = _mm256_and_si256(t_even, mask_16bit);

    // Sign extension
    let sign_bit = _mm256_and_si256(t_even_masked, _mm256_set1_epi64x(0x8000));
    let sign_extend = _mm256_cmpeq_epi64(sign_bit, _mm256_set1_epi64x(0x8000));
    let sign_fill = _mm256_and_si256(
        sign_extend,
        _mm256_set1_epi64x(0xFFFFFFFFFFFF0000u64 as i64),
    );
    let t_even_signed = _mm256_or_si256(t_even_masked, sign_fill);

    let mult_result_even = _mm256_mul_epi32(t_even_signed, vq64);
    let sub_result_even = _mm256_sub_epi64(prod64_even, mult_result_even);
    let result_even_64 = _mm256_srli_epi64(sub_result_even, 16);

    // === Odd elements [1, 3, 5, 7] ===
    let va_odd = _mm256_srli_si256(va, 4);
    let vb_odd = _mm256_srli_si256(vb, 4);
    let prod64_odd = _mm256_mul_epi32(va_odd, vb_odd);
    let t_odd = _mm256_mul_epi32(prod64_odd, vqinv);
    let t_odd_masked = _mm256_and_si256(t_odd, mask_16bit);

    let sign_bit_odd = _mm256_and_si256(t_odd_masked, _mm256_set1_epi64x(0x8000));
    let sign_extend_odd = _mm256_cmpeq_epi64(sign_bit_odd, _mm256_set1_epi64x(0x8000));
    let sign_fill_odd = _mm256_and_si256(
        sign_extend_odd,
        _mm256_set1_epi64x(0xFFFFFFFFFFFF0000u64 as i64),
    );
    let t_odd_signed = _mm256_or_si256(t_odd_masked, sign_fill_odd);

    let mult_result_odd = _mm256_mul_epi32(t_odd_signed, vq64);
    let sub_result_odd = _mm256_sub_epi64(prod64_odd, mult_result_odd);
    let result_odd_64 = _mm256_srli_epi64(sub_result_odd, 16);

    // opt: pack using shuffle instead of store/load
    let even_lo = _mm256_castsi256_si128(result_even_64);
    let even_hi = _mm256_extracti128_si256(result_even_64, 1);
    let odd_lo = _mm256_castsi256_si128(result_odd_64);
    let odd_hi = _mm256_extracti128_si256(result_odd_64, 1);

    let even_lo_32 = _mm_shuffle_epi32(even_lo, 0b11_01_10_00);
    let even_hi_32 = _mm_shuffle_epi32(even_hi, 0b11_01_10_00);
    let odd_lo_32 = _mm_shuffle_epi32(odd_lo, 0b11_01_10_00);
    let odd_hi_32 = _mm_shuffle_epi32(odd_hi, 0b11_01_10_00);

    let lo_result = _mm_unpacklo_epi32(even_lo_32, odd_lo_32);
    let hi_result = _mm_unpacklo_epi32(even_hi_32, odd_hi_32);

    let result = _mm256_set_m128i(hi_result, lo_result);

    let mut output = [0i32; 8];
    _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
    output
}

#[target_feature(enable = "avx2")]
pub unsafe fn simd_barrett_reduce_8(a: &[i32; 8]) -> [i32; 8] {
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let v: i64 = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
    let vv = _mm256_set1_epi64x(v);

    // === Indexes [0, 2, 4, 6] ===
    let va_lower = _mm256_castsi256_si128(va);
    let va_02_64 = _mm256_cvtepi32_epi64(va_lower);
    let prod_02 = _mm256_mul_epi32(vv, va_02_64);
    let t_02_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_02, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // === Indexes [1, 3] ===
    let va_shifted = _mm256_srli_si256(va, 4);
    let va_shifted_lower = _mm256_castsi256_si128(va_shifted);
    let va_13_64 = _mm256_cvtepi32_epi64(va_shifted_lower);
    let prod_13 = _mm256_mul_epi32(vv, va_13_64);
    let t_13_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_13, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // === Indexes [4, 6] ===
    let va_upper = _mm256_extracti128_si256(va, 1);
    let va_46_64 = _mm256_cvtepi32_epi64(va_upper);
    let prod_46 = _mm256_mul_epi32(vv, va_46_64);
    let t_46_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_46, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // === Indexes [5, 7] ===
    let va_upper_shifted = _mm_srli_si128(va_upper, 4);
    let va_57_64 = _mm256_cvtepi32_epi64(va_upper_shifted);
    let prod_57 = _mm256_mul_epi32(vv, va_57_64);
    let t_57_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_57, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    let mut t_02 = [0i64; 4];
    let mut t_13 = [0i64; 4];
    let mut t_46 = [0i64; 4];
    let mut t_57 = [0i64; 4];
    _mm256_storeu_si256(t_02.as_mut_ptr() as *mut __m256i, t_02_64);
    _mm256_storeu_si256(t_13.as_mut_ptr() as *mut __m256i, t_13_64);
    _mm256_storeu_si256(t_46.as_mut_ptr() as *mut __m256i, t_46_64);
    _mm256_storeu_si256(t_57.as_mut_ptr() as *mut __m256i, t_57_64);

    let mut t_values = [0i32; 8];
    t_values[0] = t_02[0] as i32;
    t_values[1] = t_13[0] as i32;
    t_values[2] = t_02[2] as i32;
    t_values[3] = t_13[2] as i32;
    t_values[4] = t_46[0] as i32;
    t_values[5] = t_57[0] as i32;
    t_values[6] = t_46[2] as i32;
    t_values[7] = t_57[2] as i32;

    let mut output = [0i32; 8];
    for i in 0..8 {
        output[i] = a[i] - t_values[i] * Q;
    }
    output
}

#[target_feature(enable = "avx2")]
pub unsafe fn simd_barrett_reduce_8_register_opt(a: &[i32; 8]) -> [i32; 8] {
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let v: i64 = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
    let vv = _mm256_set1_epi64x(v);

    // === Lower 128 bits (elements 0-3) ===
    let va_lower = _mm256_castsi256_si128(va); // [a0, a1, a2, a3]

    // Elements 0, 1
    let va_01_64 = _mm256_cvtepi32_epi64(va_lower); // converts [a0, a1] -> i64
    let prod_01 = _mm256_mul_epi32(vv, va_01_64);
    let t_01_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_01, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // Elements 2, 3 - need to shift va_lower first
    let va_lower_shifted = _mm_srli_si128(va_lower, 8); // [a2, a3, 0, 0]
    let va_23_64 = _mm256_cvtepi32_epi64(va_lower_shifted); // converts [a2, a3] -> i64
    let prod_23 = _mm256_mul_epi32(vv, va_23_64);
    let t_23_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_23, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // === Upper 128 bits (elements 4-7) ===
    let va_upper = _mm256_extracti128_si256(va, 1); // [a4, a5, a6, a7]

    // Elements 4, 5
    let va_45_64 = _mm256_cvtepi32_epi64(va_upper);
    let prod_45 = _mm256_mul_epi32(vv, va_45_64);
    let t_45_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_45, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // Elements 6, 7
    let va_upper_shifted = _mm_srli_si128(va_upper, 8); // [a6, a7, 0, 0]
    let va_67_64 = _mm256_cvtepi32_epi64(va_upper_shifted);
    let prod_67 = _mm256_mul_epi32(vv, va_67_64);
    let t_67_64 = _mm256_srli_epi64(
        _mm256_add_epi64(prod_67, _mm256_set1_epi64x(1i64 << 25)),
        26,
    );

    // === Pack using shuffle ===
    // Now we have:
    // t_01_64: [t0, X, t1, X] as i64
    // t_23_64: [t2, X, t3, X] as i64
    // t_45_64: [t4, X, t5, X] as i64
    // t_67_64: [t6, X, t7, X] as i64

    let t_01_lo = _mm256_castsi256_si128(t_01_64); // [t0, t1] as i64
    let t_23_lo = _mm256_castsi256_si128(t_23_64); // [t2, t3] as i64
    let t_45_lo = _mm256_castsi256_si128(t_45_64); // [t4, t5] as i64
    let t_67_lo = _mm256_castsi256_si128(t_67_64); // [t6, t7] as i64

    // Shuffle to extract lower 32 bits from each i64
    let t_01_i32 = _mm_shuffle_epi32(t_01_lo, 0b00_00_10_00); // [t0, ?, t1, ?]
    let t_23_i32 = _mm_shuffle_epi32(t_23_lo, 0b00_00_10_00); // [t2, ?, t3, ?]
    let t_45_i32 = _mm_shuffle_epi32(t_45_lo, 0b00_00_10_00); // [t4, ?, t5, ?]
    let t_67_i32 = _mm_shuffle_epi32(t_67_lo, 0b00_00_10_00); // [t6, ?, t7, ?]

    // Unpack to interleave
    let t_0123 = _mm_unpacklo_epi32(t_01_i32, t_23_i32); // [t0, t2, t1, t3]
    let t_4567 = _mm_unpacklo_epi32(t_45_i32, t_67_i32); // [t4, t6, t5, t7]

    // Need one more shuffle to get correct order [t0, t1, t2, t3]
    let t_0123_fixed = _mm_shuffle_epi32(t_0123, 0b11_01_10_00); // [t0, t1, t2, t3]
    let t_4567_fixed = _mm_shuffle_epi32(t_4567, 0b11_01_10_00); // [t4, t5, t6, t7]

    let t_vec = _mm256_set_m128i(t_4567_fixed, t_0123_fixed);

    // Compute: a[i] - t[i] * Q
    let vq = _mm256_set1_epi32(Q);
    let t_mul_q = _mm256_mullo_epi32(t_vec, vq);
    let result = _mm256_sub_epi32(va, t_mul_q);

    let mut output = [0i32; 8];
    _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_opt_operations() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping SIMD tests");
            return;
        }

        let a = [10, 20, 30, 40, 50, 60, 70, 80];
        let b = [5, 10, 15, 20, 25, 30, 35, 40];

        unsafe {
            // Test add
            let result_add = simd_mod_add_8(&a, &b);
            for i in 0..8 {
                let expected = mod_add(a[i], b[i]);
                assert_eq!(result_add[i], expected, "SIMD add opt failed at {}", i);
            }

            // Test sub
            let result_sub = simd_mod_sub_8(&a, &b);
            for i in 0..8 {
                let expected = mod_sub(a[i], b[i]);
                assert_eq!(result_sub[i], expected, "SIMD sub opt failed at {}", i);
            }

            // Test mul
            let result_mul = simd_mod_mul_8(&a, &b);
            for i in 0..8 {
                let expected = mod_mul(a[i], b[i]);
                assert_eq!(result_mul[i], expected, "SIMD mul opt failed at {}", i);
            }

            // Test barrett
            let large = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];
            let result_barrett = simd_barrett_reduce_8(&large);
            for i in 0..8 {
                let expected = barrett_reduce(large[i]);
                assert_eq!(
                    result_barrett[i], expected,
                    "SIMD barrett opt failed at {}",
                    i
                );
            }
        }

        println!("all optimized SIMD operations passed");
    }
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_barrett_versions_match() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test");
            return;
        }

        let large = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];

        unsafe {
            let result_v1 = simd_barrett_reduce_8(&large);
            let result_v2 = simd_barrett_reduce_8_register_opt(&large);

            assert_eq!(
                result_v1, result_v2,
                "Barrett reduction versions produce different results"
            );
        }

        println!("Both Barrett reduction versions match");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn debug_barrett_packing() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test");
            return;
        }

        let large = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];

        unsafe {
            let va = _mm256_loadu_si256(large.as_ptr() as *const __m256i);
            let v: i64 = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
            let vv = _mm256_set1_epi64x(v);

            // Compute t values
            let va_lower = _mm256_castsi256_si128(va);
            let va_02_64 = _mm256_cvtepi32_epi64(va_lower);
            let prod_02 = _mm256_mul_epi32(vv, va_02_64);
            let t_02_64 = _mm256_srli_epi64(
                _mm256_add_epi64(prod_02, _mm256_set1_epi64x(1i64 << 25)),
                26,
            );

            let va_shifted = _mm256_srli_si256(va, 4);
            let va_shifted_lower = _mm256_castsi256_si128(va_shifted);
            let va_13_64 = _mm256_cvtepi32_epi64(va_shifted_lower);
            let prod_13 = _mm256_mul_epi32(vv, va_13_64);
            let t_13_64 = _mm256_srli_epi64(
                _mm256_add_epi64(prod_13, _mm256_set1_epi64x(1i64 << 25)),
                26,
            );

            // Store and print
            let mut t_02 = [0i64; 4];
            let mut t_13 = [0i64; 4];
            _mm256_storeu_si256(t_02.as_mut_ptr() as *mut __m256i, t_02_64);
            _mm256_storeu_si256(t_13.as_mut_ptr() as *mut __m256i, t_13_64);

            println!("t_02 (as i64): {:?}", t_02);
            println!("t_13 (as i64): {:?}", t_13);
            println!(
                "t_02 (as i32): [{}, {}, {}, {}]",
                t_02[0] as i32,
                (t_02[0] >> 32) as i32,
                t_02[2] as i32,
                (t_02[2] >> 32) as i32
            );
            println!(
                "t_13 (as i32): [{}, {}, {}, {}]",
                t_13[0] as i32,
                (t_13[0] >> 32) as i32,
                t_13[2] as i32,
                (t_13[2] >> 32) as i32
            );

            // Test shuffle approach
            let t_02_lo = _mm256_castsi256_si128(t_02_64);
            let t_13_lo = _mm256_castsi256_si128(t_13_64);

            let t_02_i32 = _mm_shuffle_epi32(t_02_lo, 0b00_00_10_00);
            let t_13_i32 = _mm_shuffle_epi32(t_13_lo, 0b00_00_10_00);

            let mut shuffled_02 = [0i32; 4];
            let mut shuffled_13 = [0i32; 4];
            _mm_storeu_si128(shuffled_02.as_mut_ptr() as *mut __m128i, t_02_i32);
            _mm_storeu_si128(shuffled_13.as_mut_ptr() as *mut __m128i, t_13_i32);

            println!("After shuffle t_02: {:?}", shuffled_02);
            println!("After shuffle t_13: {:?}", shuffled_13);

            let t_0123 = _mm_unpacklo_epi32(t_02_i32, t_13_i32);
            let mut unpacked = [0i32; 4];
            _mm_storeu_si128(unpacked.as_mut_ptr() as *mut __m128i, t_0123);
            println!("After unpack [t0,t1,t2,t3]: {:?}", unpacked);

            println!(
                "\nExpected: [t0={}, t1={}, t2={}, t3={}]",
                t_02[0] as i32, t_13[0] as i32, t_02[2] as i32, t_13[2] as i32
            );
        }
    }
}
