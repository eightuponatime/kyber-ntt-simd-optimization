//! Демонстрация: скаляр vs inline asm vs intrinsics В ЦИКЛЕ
//!
//! Задача: обрезать каждый элемент массива i32 в диапазон [0, 255]
//! (clamp). Простейшая операция, но в цикле видна разница подходов.
//!
//! Почему intrinsics быстрее inline asm в цикле:
//!   - LLVM выносит broadcast констант (0 и 255) из цикла
//!   - LLVM оптимизирует распределение регистров глобально
//!   - Inline asm — чёрный ящик: константы грузятся каждую итерацию

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================
// 1. Скалярная
// ============================================================

#[inline(never)]
pub fn clamp_scalar(data: &mut [i32]) {
    for x in data.iter_mut() {
        if *x < 0 { *x = 0; }
        if *x > 255 { *x = 255; }
    }
}

// ============================================================
// 2. Inline ASM в цикле
// ============================================================

/// Каждая итерация — отдельный asm-блок.
/// LLVM не видит что константы (0 и 255) одинаковые между итерациями
/// и не может вынести vpbroadcastd из цикла.
#[inline(never)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn clamp_inline_asm(data: &mut [i32]) {
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 8);
        let zero: i32 = 0;
        let max: i32 = 255;
        core::arch::asm!(
            "vmovdqu {v}, [{ptr}]",
            "vpbroadcastd {vmin}, dword ptr [{zero}]",
            "vpbroadcastd {vmax}, dword ptr [{max}]",
            "vpmaxsd {v}, {v}, {vmin}",
            "vpminsd {v}, {v}, {vmax}",
            "vmovdqu [{ptr}], {v}",
            ptr = in(reg) ptr,
            zero = in(reg) &zero,
            max = in(reg) &max,
            v = out(ymm_reg) _,
            vmin = out(ymm_reg) _,
            vmax = out(ymm_reg) _,
            options(nostack),
        );
    }
    // остаток скалярно
    let done = chunks * 8;
    for x in data[done..].iter_mut() {
        if *x < 0 { *x = 0; }
        if *x > 255 { *x = 255; }
    }
}

// ============================================================
// 3. Intrinsics в цикле
// ============================================================

/// Broadcast констант ДО цикла — LLVM держит их в регистрах.
/// В цикле только: load → max → min → store. Минимум инструкций.
#[inline(never)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn clamp_intrinsics(data: &mut [i32]) {
    let vmin = _mm256_set1_epi32(0);
    let vmax = _mm256_set1_epi32(255);

    let chunks = data.len() / 8;
    for i in 0..chunks {
        let ptr = data.as_mut_ptr().add(i * 8);
        let mut v = _mm256_loadu_si256(ptr as *const __m256i);
        v = _mm256_max_epi32(v, vmin);
        v = _mm256_min_epi32(v, vmax);
        _mm256_storeu_si256(ptr as *mut __m256i, v);
    }
    // остаток
    let done = chunks * 8;
    for x in data[done..].iter_mut() {
        if *x < 0 { *x = 0; }
        if *x > 255 { *x = 255; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data() -> Vec<i32> {
        vec![-100, -1, 0, 1, 127, 255, 256, 1000,
             -50, 200, 300, 128, 0, 255, -999, 42]
    }

    fn expected() -> Vec<i32> {
        vec![0, 0, 0, 1, 127, 255, 255, 255,
             0, 200, 255, 128, 0, 255, 0, 42]
    }

    #[test]
    fn test_scalar() {
        let mut d = test_data();
        clamp_scalar(&mut d);
        assert_eq!(d, expected());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_all_match() {
        if !is_x86_feature_detected!("avx2") { return; }

        let mut d_asm = test_data();
        let mut d_intr = test_data();

        unsafe {
            clamp_inline_asm(&mut d_asm);
            clamp_intrinsics(&mut d_intr);
        }

        assert_eq!(d_asm, expected(), "asm != expected");
        assert_eq!(d_intr, expected(), "intrinsics != expected");
    }
}