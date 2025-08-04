//! avx2_cosine.rs  –  no external deps, no build.rs
//! -------------------------------------------------
//! compile with:  RUSTFLAGS="-C target-cpu=native"  cargo build --release
//! or spell the features explicitly:
//! RUSTFLAGS="-C target-feature=+avx2,+fma"         cargo build --release

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
pub fn norm2_f32(v: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: we just checked the CPU supports the instructions.
            let ssq = unsafe { sumsquares_f32_avx2(v.as_ptr(), v.len()) };
            return ssq.sqrt();
        }
    }
    // portable path
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity with *pre-computed* norms.
/// Falls back to scalar multiply-add when AVX2 unavailable.
pub fn cosine_f32_opt(a: &[f32], b: &[f32], a_norm: f32, b_norm: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");

    let dot = if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
        print!("Using AVX-512 dot product... ");
        unsafe { dot_f32_avx512(a.as_ptr(), b.as_ptr(), a.len()) }
    } else {
        #[cfg(all(target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                print!("Using AVX2 dot product... ");
                // SAFETY: runtime detection above.
                unsafe { dot_f32_avx2(a.as_ptr(), b.as_ptr(), a.len()) }
            } else {
                a.iter().zip(b).map(|(&x, &y)| x * y).sum()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            a.iter().zip(b).map(|(&x, &y)| x * y).sum()
        }
    };



    dot / (a_norm * b_norm)
}

/// Build an *r × r* cosine-similarity matrix (row-major, symmetric).
///
/// * `matrix` – flat buffer holding *r* row-vectors, each of length *c*
/// * `r`      – number of rows / vectors
/// * `c`      – dimensionality of each vector
pub fn cosine_f32_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c);

    // pre-compute ‖row‖₂ once
    let norms: Vec<f32> = (0..r)
        .map(|i| norm2_f32(&matrix[i * c..(i + 1) * c]))
        .collect();

    let mut result = vec![0.0f32; r * r];

    // upper triangle (including diagonal)
    for i in 0..r {
        result[i * r + i] = 1.0;
        let a = &matrix[i * c..(i + 1) * c];

        for j in (i + 1)..r {
            let b = &matrix[j * c..(j + 1) * c];
            let sim = cosine_f32_opt(a, b, norms[i], norms[j]);
            result[i * r + j] = sim;
            result[j * r + i] = sim; // mirror
        }
    }
    result
}

/// Dot-product of `len` f32s with AVX-512 + FMA.  
/// Falls back to AVX2 / scalar at call-site.
#[target_feature(enable = "avx512f,fma")]
unsafe fn dot_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut acc = _mm512_setzero_ps();

    // 16-float chunks
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.add(i));
        let vb = _mm512_loadu_ps(b.add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += 16;
    }

    // Tail: masked load handles 0-15 remaining elements
    let rem = (len - i) as i32;
    if rem > 0 {
        let mask: __mmask16 = (!0u16) >> (16 - rem);
        let va = _mm512_maskz_loadu_ps(mask, a.add(i));
        let vb = _mm512_maskz_loadu_ps(mask, b.add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
    }

    // Horizontal add (portable across stable/nightly)
    _mm512_reduce_add_ps(acc)
}
