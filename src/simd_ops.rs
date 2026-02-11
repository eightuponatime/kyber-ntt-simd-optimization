//! shared SIMD operations for modular arithmetic

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::params::Q;

/// SIMD modular addition: (a + b) mod Q for 8 elements
#[target_feature(enable = "avx2")]
pub unsafe fn simd_mod_add_8(a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
    unsafe {
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
}

/// SIMD modular subtraction: (a - b) mod Q for 8 elements
#[target_feature(enable = "avx2")]
pub unsafe fn simd_mod_sub_8(a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
    unsafe {
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
}

/// SIMD modular multiplication: (a * b) mod Q for 8 elements using Montgomery
#[target_feature(enable = "avx2")]
pub unsafe fn simd_mod_mul_8(a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
    unsafe {
        let mut output = [0i32; 8];
        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);

        // === Even elements [0, 2, 4, 6] ===
        let prod64_even = _mm256_mul_epi32(va, vb);
        let vqinv = _mm256_set1_epi64x(-3327);
        let t_even = _mm256_mul_epi32(prod64_even, vqinv);
        let mask_16bit = _mm256_set1_epi64x(0xFFFF);
        let t_even_masked = _mm256_and_si256(t_even, mask_16bit);

        let sign_bit = _mm256_and_si256(t_even_masked, _mm256_set1_epi64x(0x8000));
        let sign_extend = _mm256_cmpeq_epi64(sign_bit, _mm256_set1_epi64x(0x8000));
        let sign_fill = _mm256_and_si256(
            sign_extend,
            _mm256_set1_epi64x(0xFFFFFFFFFFFF0000u64 as i64),
        );
        let t_even_signed = _mm256_or_si256(t_even_masked, sign_fill);

        let vq64 = _mm256_set1_epi64x(Q as i64);
        let mult_result = _mm256_mul_epi32(t_even_signed, vq64);
        let sub_result = _mm256_sub_epi64(prod64_even, mult_result);
        let result_even_64 = _mm256_srli_epi64(sub_result, 16);

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

        // === Pack results ===
        let mut temp_even = [0i64; 4];
        let mut temp_odd = [0i64; 4];
        _mm256_storeu_si256(temp_even.as_mut_ptr() as *mut __m256i, result_even_64);
        _mm256_storeu_si256(temp_odd.as_mut_ptr() as *mut __m256i, result_odd_64);

        output[0] = temp_even[0] as i32;
        output[1] = temp_odd[0] as i32;
        output[2] = temp_even[1] as i32;
        output[3] = temp_odd[1] as i32;
        output[4] = temp_even[2] as i32;
        output[5] = temp_odd[2] as i32;
        output[6] = temp_even[3] as i32;
        output[7] = temp_odd[3] as i32;
        output
    }
}

/// SIMD Barrett reduction for 8 elements
#[target_feature(enable = "avx2")]
pub unsafe fn simd_barrett_reduce_8(a: &[i32; 8]) -> [i32; 8] {
    unsafe {
        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let v: i64 = ((1i64 << 26) + (Q as i64 / 2)) / Q as i64;
        let vv = _mm256_set1_epi64x(v);

        // === Indexes [0, 2] ===
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

        // Collecting results
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_operations() {
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
                assert_eq!(result_add[i], expected, "SIMD add failed at {}", i);
            }

            // Test sub
            let result_sub = simd_mod_sub_8(&a, &b);
            for i in 0..8 {
                let expected = mod_sub(a[i], b[i]);
                assert_eq!(result_sub[i], expected, "SIMD sub failed at {}", i);
            }

            // Test mul
            let result_mul = simd_mod_mul_8(&a, &b);
            for i in 0..8 {
                let expected = mod_mul(a[i], b[i]);
                assert_eq!(result_mul[i], expected, "SIMD mul failed at {}", i);
            }

            // Test barrett
            let large = [6700, 7000, 8000, 9000, 10000, 5000, 4000, 3500];
            let result_barrett = simd_barrett_reduce_8(&large);
            for i in 0..8 {
                let expected = barrett_reduce(large[i]);
                assert_eq!(result_barrett[i], expected, "SIMD barrett failed at {}", i);
            }
        }
    }
}
