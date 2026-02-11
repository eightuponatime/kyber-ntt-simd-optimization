use crate::params::N;

pub trait NTT {
    /// forward NTT transform
    fn forward(&self, a: &mut [i32; N]);
    
    /// inverse NTT transform
    fn inverse(&self, a: &mut [i32; N]);
    
    /// roundtrip
    fn ntt_inv_ntt(&self, a: &mut [i32; N]) {
        self.forward(a);
        self.inverse(a);
    }
    
    fn name(&self) -> &'static str;
}
