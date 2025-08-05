// cosine_mat.rs  —  `cargo run --release` (see build flags at bottom)
use rayon::prelude::*;
use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    arch::x86_64::*,
    ptr::NonNull,
};

/// ---------- 32-byte-aligned heap buffer ------------------------------------
struct AlignedBuf {
    ptr: NonNull<f32>,
    len: usize,
}
impl AlignedBuf {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len * 4, 32).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        Self { ptr: NonNull::new(ptr).expect("alloc"), len }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len * 4, 32).unwrap();
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
    }
}
impl std::ops::Deref for AlignedBuf { type Target = [f32];
    fn deref(&self) -> &Self::Target { unsafe {
        std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
}
impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut Self::Target { unsafe {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) } }
}

/// ---------- scalar helpers --------------------------------------------------
#[inline] fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x,&y)| x*y).sum()
}
#[inline] fn norm2_scalar(x: &[f32]) -> f32 {
    x.iter().map(|&v| v*v).sum::<f32>().sqrt()
}

/// ---------- AVX2 kernel: 4 accumulators, masked tail ------------------------
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_4acc(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut a0=_mm256_setzero_ps(); let mut a1=a0; let mut a2=a0; let mut a3=a0;

    while i + 32 <= len {
        let va0=_mm256_load_ps(a.add(i     )); let vb0=_mm256_load_ps(b.add(i     ));
        let va1=_mm256_load_ps(a.add(i +  8)); let vb1=_mm256_load_ps(b.add(i +  8));
        let va2=_mm256_load_ps(a.add(i + 16)); let vb2=_mm256_load_ps(b.add(i + 16));
        let va3=_mm256_load_ps(a.add(i + 24)); let vb3=_mm256_load_ps(b.add(i + 24));

        a0=_mm256_fmadd_ps(va0,vb0,a0); a1=_mm256_fmadd_ps(va1,vb1,a1);
        a2=_mm256_fmadd_ps(va2,vb2,a2); a3=_mm256_fmadd_ps(va3,vb3,a3);
        i += 32;
    }
    let mut acc = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));

    // 0–31-element tail
    let rem = len - i;
    if rem!=0 {
        let mask: __mmask8 = ((1u16 << rem) - 1) as u8;
        // loadu_ps_mask is nightly; emulate with masked load
        let va = _mm256_maskz_loadu_ps(mask as i32, a.add(i));
        let vb = _mm256_maskz_loadu_ps(mask as i32, b.add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_add_ps(acc,
                                                       _mm256_permute2f128_ps(acc, acc, 1))))
        + {  // reduce 128-vector
        let s = _mm_add_ps(_mm256_castps256_ps128(acc),
                           _mm_movehl_ps(_mm256_castps256_ps128(acc), _mm256_castps256_ps128(acc)));
        let s = _mm_add_ps(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }
}

/// ---------- normalise rows (parallel, AVX2 if possible) ---------------------
fn normalise_rows(src: &[f32], r: usize, c: usize, dst: &mut [f32], use_avx: bool)
{
    (0..r).into_par_iter().for_each(|i| {
        let row_src = &src[i*c .. (i+1)*c];
        let row_dst = &mut dst[i*c .. (i+1)*c];

        let norm = if use_avx {
            unsafe { dot_f32_avx2_4acc(row_src.as_ptr(), row_src.as_ptr(), c).sqrt() }
        } else { norm2_scalar(row_src) };

        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for (d, &s) in row_dst.iter_mut().zip(row_src) { *d = s * inv; }
    });
}

/// ---------- public API ------------------------------------------------------
pub fn cosine_similarity_matrix(matrix: &[f32], r: usize, c: usize) -> AlignedBuf {
    assert_eq!(matrix.len(), r * c);

    let use_avx = cfg!(target_feature="avx2") ||
        std::is_x86_feature_detected!("avx2") &&
            std::is_x86_feature_detected!("fma");

    // 1. copy + L2-normalise
    let mut normed = AlignedBuf::new(r*c);
    normalise_rows(matrix, r, c, &mut normed, use_avx);

    // 2. allocate result (row-major, symmetric)
    let mut sim = AlignedBuf::new(r*r);
    const BLOCK: usize = 32;          // better load-balancing
    (0..r).into_par_iter().step_by(BLOCK).for_each(|base| {
        let end = (base + BLOCK).min(r);
        for i in base..end {
            let row_i = &normed[i*c .. (i+1)*c];
            let sim_row = &mut sim[i*r .. (i+1)*r];
            sim_row[i] = 1.0;
            for j in (i+1)..r {
                let row_j = &normed[j*c .. (j+1)*c];
                let dot = if use_avx {
                    unsafe { dot_f32_avx2_4acc(row_i.as_ptr(), row_j.as_ptr(), c) }
                } else { dot_scalar(row_i, row_j) };
                sim_row[j] = dot;
                sim[j*r + i] = dot;            // mirror
            }
        }
    });
    sim
}

// -------- tiny demo ----------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn check_small() {
        let m = vec![1.,2.,3.,  0.,1.,0.,  1.,0.,0.]; // 3×3
        let s = cosine_similarity_matrix(&m, 3, 3);
        let want = [1.0,0.26726124,0.26726124,
            0.26726124,1.0,0.0,
            0.26726124,0.0,1.0];
        for (a,b) in s.iter().zip(want) { assert!((a-b).abs()<1e-6); }
    }
}
