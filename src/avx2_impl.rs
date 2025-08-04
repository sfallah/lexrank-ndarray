//! avx2_cosine.rs  –  no external deps, no build.rs
//! -------------------------------------------------
//! compile with:  RUSTFLAGS="-C target-cpu=native"  cargo build --release
//! or spell the features explicitly:
//! RUSTFLAGS="-C target-feature=+avx2,+fma"         cargo build --release

use rayon::prelude::*;
use std::arch::x86_64::*;

// ---------- low-level helpers ------------------------------------------------

/// Horizontal add of the eight lanes in `v`.
#[inline(always)]
unsafe fn hsum_avx2_ps(v: __m256) -> f32 {
    // (a,b,c,d,e,f,g,h) → (a+b) + (c+d) + (e+f) + (g+h)
    let hi = _mm256_extractf128_ps(v, 1); // upper 128
    let lo = _mm256_castps256_ps128(v); // lower 128
    let sum128 = _mm_add_ps(lo, hi); // 4-lane sums

    // 4-lane -> 2-lane -> 1
    let shuf = _mm_movehdup_ps(sum128); // (b,d, f,h)
    let sum2 = _mm_add_ps(sum128, shuf); // (a+b, c+d, e+f, g+h)
    let hi64 = _mm_movehl_ps(shuf, sum2); // (c+d, g+h)
    let sum1 = _mm_add_ss(sum2, hi64); // (total, …)
    _mm_cvtss_f32(sum1)
}

/// Fused multiply-add dot-product for *exactly* `len` elements.
/// `#[target_feature]` tells LLVM it can freely use AVX/FMA here.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        acc = _mm256_fmadd_ps(va, vb, acc); // acc += va * vb
        i += 8;
    }

    let mut sum = hsum_avx2_ps(acc);

    // tail
    while i < len {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }
    sum
}

/// Sum of squares for `len` elements – building block for ‖·‖₂.
#[target_feature(enable = "avx2,fma")]
unsafe fn sumsquares_f32_avx2(p: *const f32, len: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();

    while i + 8 <= len {
        let v = _mm256_loadu_ps(p.add(i));
        acc = _mm256_fmadd_ps(v, v, acc); // acc += v * v
        i += 8;
    }

    let mut sum = hsum_avx2_ps(acc);

    while i < len {
        let x = *p.add(i);
        sum += x * x;
        i += 1;
    }
    sum
}

// ---------- public, safe API -------------------------------------------------

/// L2-norm using AVX2 if available, else scalar fallback.
#[cfg(all(target_arch = "x86_64"))]
#[inline(always)]
pub fn norm2_f32(v: &[f32]) -> f32 {
    {
        // SAFETY: we just checked the CPU supports the instructions.
        let ssq = unsafe { sumsquares_f32_avx2(v.as_ptr(), v.len()) };
        return ssq.sqrt();
    }
}

/// Cosine similarity with *pre-computed* norms.
/// Falls back to scalar multiply-add when AVX2 unavailable.
///
#[inline]
#[cfg(all(target_arch = "x86_64"))]
pub fn cosine_f32_opt(a: &[f32], b: &[f32], a_norm: f32, b_norm: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");

    let dot = { unsafe { dot_f32_avx2(a.as_ptr(), b.as_ptr(), a.len()) } };

    dot / (a_norm * b_norm)
}

/// Build an *r × r* cosine-similarity matrix (row-major, symmetric).
///
/// * `matrix` – flat buffer holding *r* row-vectors, each of length *c*
/// * `r`      – number of rows / vectors
/// * `c`      – dimensionality of each vector
pub fn cosine_f32_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c);

    // ❶ allocate the square result (row-major)
    let mut result = vec![0.0f32; r * r];

    let mut norms = vec![0.0f32; r];
    norms.par_iter_mut().enumerate().for_each(|(i, norm)| {
        // -- row i norm --
        let row_i = &matrix[i * c..(i + 1) * c];
        *norm = norm2_f32(row_i); // compute row i norm
    });

    // ❷ process each *row slice* of `result` in parallel
    //
    // `par_chunks_mut(r)` splits the buffer into disjoint mutable chunks,
    // one per row, so every thread owns a unique region and no locks are needed.
    result
        .par_chunks_mut(r) // &mut [f32] for one row
        .enumerate() // (i, row_i)
        .for_each(|(i, row_i)| {
            // -- diagonal --
            row_i[i] = 1.0;

            // -- upper triangle: j > i --
            let a = &matrix[i * c..(i + 1) * c];
            for j in (i + 1)..r {
                let b = &matrix[j * c..(j + 1) * c];
                row_i[j] = cosine_f32_opt(a, b, norms[i], norms[j]); // upper-tri entry
            }
        });

    // ❸ mirror the upper triangle → lower triangle (cheap, serial)
    for i in 0..r {
        for j in (i + 1)..r {
            let sim = result[i * r + j];
            result[j * r + i] = sim;
        }
    }

    result
}
