// cosine_matrix_avx2.rs -------------------------------------------------------
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]

use rayon::prelude::*;
use std::arch::x86_64::*;

const BLOCK: usize = 8; // number of rows per task - tune if needed

// ---------------------------------------------------------------------------
// 1.  Portable helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[inline(always)]
fn norm_scalar(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// 2.  AVX2 / FMA kernels
// ---------------------------------------------------------------------------

/// Horizontal add of eight lanes.
#[inline(always)]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi); // 4-lane sums
    let shuf = _mm_movehdup_ps(sum128); // (b,d,f,h)
    let sum64 = _mm_add_ps(sum128, shuf); // (a+b,c+d,e+f,g+h)
    let hi64 = _mm_movehl_ps(shuf, sum64); // (c+d,g+h)
    let sum32 = _mm_add_ps(sum64, hi64); // total
    _mm_cvtss_f32(sum32)
}

/// 4-way-unrolled FMA dot product.  
/// *Caller guarantees* `len >= 32`, otherwise use scalar path.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_unrolled(a: *const f32, b: *const f32, len: usize) -> f32 {
    debug_assert!(len >= 32);

    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= len {
        // 32 floats = 4 × 256-bit registers
        let va0 = _mm256_loadu_ps(a.add(i));
        let vb0 = _mm256_loadu_ps(b.add(i));
        let va1 = _mm256_loadu_ps(a.add(i + 8));
        let vb1 = _mm256_loadu_ps(b.add(i + 8));
        let va2 = _mm256_loadu_ps(a.add(i + 16));
        let vb2 = _mm256_loadu_ps(b.add(i + 16));
        let va3 = _mm256_loadu_ps(a.add(i + 24));
        let vb3 = _mm256_loadu_ps(b.add(i + 24));

        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        i += 32;
    }

    // Fold four accumulators → one.
    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut sum = hsum256_ps(acc);

    // 0–31 element tail
    while i < len {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

/// FMA sums-of-squares  (needed for normalisation)
#[target_feature(enable = "avx2,fma")]
unsafe fn sumsquares_f32_avx2_unrolled(p: *const f32, len: usize) -> f32 {
    debug_assert!(len >= 32);

    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(p.add(i));
        let v1 = _mm256_loadu_ps(p.add(i + 8));
        let v2 = _mm256_loadu_ps(p.add(i + 16));
        let v3 = _mm256_loadu_ps(p.add(i + 24));

        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
        i += 32;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut sum = hsum256_ps(acc);

    while i < len {
        let x = *p.add(i);
        sum += x * x;
        i += 1;
    }
    sum
}

// ---------------------------------------------------------------------------
// 3.  Public API
// ---------------------------------------------------------------------------

/// **Normalises** each row of `src` (size `r×c` row-major) into a new buffer
/// that is at least 32-byte aligned so the SIMD loads can use unaligned
/// instructions safely.
///
/// Returns `(buffer, stride)` where `stride == c`.
fn normalize_rows_l2(src: &[f32], r: usize, c: usize, use_avx: bool) -> Vec<f32> {
    assert_eq!(src.len(), r * c);

    // Allocate with alignment.  On nightly you could use `Vec::with_capacity_in`
    // and a 32-byte aligned allocator; on stable we rely on the fact that
    // the global allocator gives ≥ 16 B alignment and `_mm256_loadu_ps` is cheap.
    let mut dst = vec![0f32; r * c];

    dst.par_chunks_mut(BLOCK * c)
        .enumerate()
        .for_each(|(i, chunk)| {
            let start = i * BLOCK * c;
            let end = start + chunk.len();
            let row_count = (end - start) / c;

            for j in 0..row_count {
                let row_start = start + j * c;
                let row_end = row_start + c;
                let row_src = &src[row_start..row_end];
                let row_dst = &mut chunk[j * c..(j + 1) * c];

                // Compute L2 norm
                let norm = if use_avx && c >= 32 {
                    unsafe { sumsquares_f32_avx2_unrolled(row_src.as_ptr(), c) }.sqrt()
                } else {
                    norm_scalar(row_src)
                };

                // Avoid div‐by-zero
                let scale = if norm > 0.0 { 1.0 / norm } else { 0.0 };
                let scale_vec = [scale; 8];

                if use_avx && c >= 32 {
                    let scale_m256 = unsafe { _mm256_broadcast_ss(&scale) };
                    let mut k = 0usize;
                    // 8-way vector multiply
                    while k + 8 <= c {
                        unsafe {
                            let v = _mm256_loadu_ps(row_src.as_ptr().add(k));
                            let sv = _mm256_mul_ps(v, scale_m256);
                            _mm256_storeu_ps(row_dst.as_mut_ptr().add(k), sv);
                        }
                        k += 8;
                    }
                    // tail
                    while k < c {
                        row_dst[k] = row_src[k] * scale;
                        k += 1;
                    }
                } else {
                    // scalar fallback
                    for k in 0..c {
                        row_dst[k] = row_src[k] * scale;
                    }
                }
            }
        });

    dst
}

/// Compute the **r × r** cosine-similarity matrix of the `r` row-vectors in
/// `matrix` (`row-major`, each row length = `c`).  
/// *The input is left untouched.*
pub fn cosine_similarity_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c);

    // ----------  one-time SIMD capability check ----------
    let use_avx = cfg!(target_feature = "avx2")
        || std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");

    // ---------- 1.  L2-normalise source into a new buffer ----------
    let normed = normalize_rows_l2(matrix, r, c, use_avx);

    // ---------- 2.  Allocate similarity matrix ----------
    let mut sim = vec![0.0f32; r * r];

    sim.par_chunks_mut(r)
        .enumerate()
        .for_each(|(i, sim_chunk)| {
            let row_i = &normed[i * c..(i + 1) * c];
            sim_chunk[i] = 1.0;
            sim_chunk[(i + 1)..r]
                .par_chunks_mut(BLOCK)
                .enumerate()
                .for_each(|(k, sim_sub_chunck)| {
                    for j in 0..sim_sub_chunck.len() {
                        let j_idx = k * BLOCK + j; // global index
                        // Compute cosine similarity for row_i and row_j
                        let row_j = &normed[(i + j_idx + 1) * c..(i + j_idx + 2) * c];
                        let dot = if use_avx && c >= 32 {
                            unsafe { dot_f32_avx2_unrolled(row_i.as_ptr(), row_j.as_ptr(), c) }
                        } else {
                            dot_scalar(row_i, row_j)
                        };
                        sim_sub_chunck[j] = dot; // upper triangle
                    }
                }); // diagonal
        });

    // ---------- 3.  Copy upper triangle to lower triangle ----------
    for i in 0..r {
        for j in (i + 1)..r {
            sim[j * r + i] = sim[i * r + j]; // copy
        }
    }

    sim
}

// ---------------------------------------------------------------------------
// 4.  Tiny smoke-test (run with `cargo test --release`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_with_scalar_small() {
        // two random 4-D vectors, easy to check by hand
        let m = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let sims = cosine_similarity_matrix(&m, 2, 4);
        assert!((sims[0] - 1.0).abs() < 1e-6);
        assert!((sims[3] - 1.0).abs() < 1e-6);
        assert!((sims[1] - 0.0).abs() < 1e-6);
        assert!((sims[2] - 0.0).abs() < 1e-6);
    }
}
