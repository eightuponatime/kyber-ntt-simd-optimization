//! NTT algo realization using SIMD instructions

use super::traits::NTT;
use crate::modular::{barrett_reduce, mod_add, mod_mul, mod_sub};
use crate::params::{N, Q, ZETAS};
use crate::simd_ops::{simd_barrett_reduce_8, simd_mod_add_8, simd_mod_mul_8, simd_mod_sub_8};

pub struct SimdNTT {
    twiddles: Vec<i32>,
}

impl SimdNTT {
    pub fn new() -> Self {
        let mut twiddles = Vec::with_capacity(128);
        for i in 0..128 {
            twiddles.push(ZETAS[i] as i32);
        }
        Self { twiddles }
    }

    /// SIMD-optimized forward NTT (checking AVX2 arch OUTSIDE)
    #[target_feature(enable = "avx2")]
    unsafe fn forward_simd(&self, a: &mut [i32; 256]) {
        let mut k = 1;
        let mut len = 128;

        while len >= 2 {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.twiddles[k];
                k += 1;

                let mut j = start;
                let end = start + len;

                // SIMD: processing 8 elements per time
                while j + 8 <= end {
                    let a_lower = [
                        a[j],
                        a[j + 1],
                        a[j + 2],
                        a[j + 3],
                        a[j + 4],
                        a[j + 5],
                        a[j + 6],
                        a[j + 7],
                    ];
                    let a_upper = [
                        a[j + len],
                        a[j + len + 1],
                        a[j + len + 2],
                        a[j + len + 3],
                        a[j + len + 4],
                        a[j + len + 5],
                        a[j + len + 6],
                        a[j + len + 7],
                    ];
                    let zeta_vec = [zeta; 8];

                    let t = simd_mod_mul_8(&a_upper, &zeta_vec);
                    let a_new = simd_mod_add_8(&a_lower, &t);
                    let b_new = simd_mod_sub_8(&a_lower, &t);

                    for i in 0..8 {
                        a[j + i] = a_new[i];
                        a[j + len + i] = b_new[i];
                    }

                    j += 8;
                }

                // Остаток
                while j < end {
                    let t = mod_mul(a[j + len], zeta);
                    a[j + len] = mod_sub(a[j], t);
                    a[j] = mod_add(a[j], t);
                    j += 1;
                }
            }
            len >>= 1;
        }
    }

    /// SIMD-optimized inverse NTT (also checking AVX2 arch outside)
    #[target_feature(enable = "avx2")]
    unsafe fn inverse_simd(&self, a: &mut [i32; 256]) {
        let mut k = 127;
        let mut len = 2;

        while len <= 128 {
            let mut start = 0;

            while start < N {
                let zeta = self.twiddles[k];
                k = k.saturating_sub(1);

                let mut j = start;
                let end = start + len;

                // SIMD: processing 8 elements per time
                while j + 8 <= end {
                    let a_lower = [
                        a[j],
                        a[j + 1],
                        a[j + 2],
                        a[j + 3],
                        a[j + 4],
                        a[j + 5],
                        a[j + 6],
                        a[j + 7],
                    ];
                    let a_upper = [
                        a[j + len],
                        a[j + len + 1],
                        a[j + len + 2],
                        a[j + len + 3],
                        a[j + len + 4],
                        a[j + len + 5],
                        a[j + len + 6],
                        a[j + len + 7],
                    ];
                    let zeta_vec = [zeta; 8];

                    let sum = simd_mod_add_8(&a_lower, &a_upper);
                    let a_new = simd_barrett_reduce_8(&sum);

                    let diff = simd_mod_sub_8(&a_upper, &a_lower);
                    let b_new = simd_mod_mul_8(&diff, &zeta_vec);

                    for i in 0..8 {
                        a[j + i] = a_new[i];
                        a[j + len + i] = b_new[i];
                    }

                    j += 8;
                }

                // residual
                while j < end {
                    let t = a[j];
                    a[j] = barrett_reduce(t + a[j + len]);
                    a[j + len] = mod_sub(a[j + len], t);
                    a[j + len] = mod_mul(a[j + len], zeta);
                    j += 1;
                }

                start = j + len;
            }
            len <<= 1;
        }

        // final normalization
        let f = 512;
        let f_vec = [f; 8];
        let mut i = 0;

        while i + 8 <= 256 {
            let chunk = [
                a[i],
                a[i + 1],
                a[i + 2],
                a[i + 3],
                a[i + 4],
                a[i + 5],
                a[i + 6],
                a[i + 7],
            ];
            let result = simd_mod_mul_8(&chunk, &f_vec);

            for j in 0..8 {
                a[i + j] = result[j];
                if a[i + j] < 0 {
                    a[i + j] += Q;
                }
            }
            i += 8;
        }

        // Residual
        while i < 256 {
            a[i] = mod_mul(a[i], f);
            if a[i] < 0 {
                a[i] += Q;
            }
            i += 1;
        }
    }

    /// fallback: scalar version of forward
    fn forward_scalar(&self, a: &mut [i32; 256]) {
        let mut k = 1;
        let mut len = 128;

        while len >= 2 {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.twiddles[k];
                k += 1;

                for j in start..(start + len) {
                    let t = mod_mul(a[j + len], zeta);
                    a[j + len] = mod_sub(a[j], t);
                    a[j] = mod_add(a[j], t);
                }
            }
            len >>= 1;
        }
    }

    /// fallback: scalar version of inverse
    fn inverse_scalar(&self, a: &mut [i32; 256]) {
        let mut k = 127;
        let mut len = 2;

        while len <= 128 {
            let mut start = 0;
            while start < N {
                let zeta = self.twiddles[k];
                k = k.saturating_sub(1);

                for j in start..(start + len) {
                    let t = a[j];
                    a[j] = barrett_reduce(t + a[j + len]);
                    a[j + len] = mod_sub(a[j + len], t);
                    a[j + len] = mod_mul(a[j + len], zeta);
                }

                start += 2 * len;
            }
            len <<= 1;
        }

        // Final normalization
        let f = 512;
        for x in a.iter_mut() {
            *x = mod_mul(*x, f);
            if *x < 0 {
                *x += Q;
            }
        }
    }
}

impl NTT for SimdNTT {
    fn forward(&self, a: &mut [i32; 256]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.forward_simd(a);
                }
                return;
            }
        }

        self.forward_scalar(a);
    }

    fn inverse(&self, a: &mut [i32; 256]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    self.inverse_simd(a);
                }
                return;
            }
        }

        self.inverse_scalar(a);
    }

    fn name(&self) -> &'static str {
        "SimdNTT"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation() {
        let _ = SimdNTT::new();
    }

    #[test]
    fn test_simd_roundtrip() {
        let ntt = SimdNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..8 {
            poly[i] = (i + 1) as i32;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original);
    }

    #[test]
    fn test_simd_full() {
        let ntt = SimdNTT::new();
        let mut poly = [0i32; 256];

        for i in 0..256 {
            poly[i] = (i as i32 * 13) % 3329;
        }

        let original = poly.clone();
        ntt.forward(&mut poly);
        ntt.inverse(&mut poly);

        assert_eq!(poly, original);
    }

    #[test]
    fn compare_with_basic() {
        use crate::ntt::BasicNTT;

        let basic_ntt = BasicNTT::new();
        let simd_ntt = SimdNTT::new();

        let mut poly1 = [0i32; 256];
        let mut poly2 = [0i32; 256];

        for i in 0..256 {
            let val = (i as i32 * 17) % 3329;
            poly1[i] = val;
            poly2[i] = val;
        }

        basic_ntt.forward(&mut poly1);
        simd_ntt.forward(&mut poly2);
        assert_eq!(poly1, poly2);

        basic_ntt.inverse(&mut poly1);
        simd_ntt.inverse(&mut poly2);
        assert_eq!(poly1, poly2);
    }
}
