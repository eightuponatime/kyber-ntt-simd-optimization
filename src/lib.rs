pub mod llvm_demo;
pub mod modular;
pub mod ntt;
pub mod params;
pub mod simd_ops;
pub mod simd_ops_opt;

pub use ntt::{BasicNTT, SimdNTT, SimdNttOpt, NTT};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
