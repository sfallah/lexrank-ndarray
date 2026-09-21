use crate::zero_rows_without_direction;
use anyhow::bail;
use cblas::*;
use rayon::prelude::*;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
extern crate cblas;
#[cfg(any(feature = "blas", feature = "blas-static"))]
extern crate openblas_src;

/// Apply soft-max row-wise to an `m × n` matrix stored row-major.
///
/// `weights.len()` must be `m * n`.
///
/// Returns a new `Vec<f32>` with the soft-max probabilities (same layout).
pub fn blas_softmax(weights: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(weights.len(), m * n, "dimension mismatch");

    // 1) Element-wise exp — BLAS has no exp, so we do it manually.
    let mut exp_data: Vec<f32> = weights.iter().map(|&v| v.exp()).collect();

    // 2) Row sums:  y = 1 · exp_data   (sgemm would also work, but sgemv is cheaper)
    let ones = vec![1.0_f32; n]; // column vector of 1s
    let mut row_sums = vec![0.0_f32; m]; // result vector

    unsafe {
        sgemv(
            Layout::RowMajor,
            Transpose::None,
            m as i32,      // rows of A
            n as i32,      // cols of A
            1.0,           // α
            &exp_data,     // A
            n as i32,      // leading dimension (row stride)
            &ones,         // x
            1,             // incx
            0.0,           // β
            &mut row_sums, // y
            1,             // incy
        );
    }

    // 3) Normalise each row:  exp_data[r] /= row_sums[r]
    for r in 0..m {
        let inv_sum = 1.0 / row_sums[r];
        unsafe {
            sscal(
                n as i32,               // number of elements in the row
                inv_sum,                // scale factor
                &mut exp_data[r * n..], // row start
                1,                      // stride
            );
        }
    }

    exp_data
}

/// Discretise the matrix (entries ≥ `threshold` → 1, otherwise 0) and
/// then turn it into a row-stochastic Markov matrix.
///
/// Falls back to the BLAS-powered `create_markov_matrix`.
///
/// * `weights` – flat row-major slice of length `m * n`.
pub fn blas_create_markov_matrix_discrete(
    weights: &[f32],
    m: usize,
    n: usize,
    threshold: f32,
) -> Vec<f32> {
    assert_eq!(weights.len(), m * n, "dimension mismatch");

    // 1) Threshold to {0,1}
    let mut discrete = Vec::with_capacity(weights.len());
    for &v in weights {
        discrete.push(if v >= threshold { 1.0 } else { 0.0 });
    }

    // 2) Row-normalise via the BLAS implementation we already wrote
    blas_create_markov_matrix(&discrete, m, n)
}

pub fn blas_create_markov_matrix(weights: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(weights.len(), m * n, "dimension mismatch");

    // ------- 1) Global minimum ------------------------------------------------
    let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);

    // ------- 2) If any entry ≤ 0 → use soft-max -------------------------------
    if min_val <= 0.0 {
        return blas_softmax(weights, m, n);
    }

    // ------- 3) Otherwise: row normalisation ----------------------------------
    // Copy data so we can mutate in-place.
    let mut out = weights.to_vec();

    // a) Row sums via y = 1 · out   (Level-2 BLAS: SGEMV)
    let ones = vec![1.0_f32; n];
    let mut row_sums = vec![0.0_f32; m];
    unsafe {
        sgemv(
            Layout::RowMajor,
            Transpose::None,
            m as i32,      // rows
            n as i32,      // cols
            1.0,           // α
            &out,          // A
            n as i32,      // lda
            &ones,         // x
            1,             // incx
            0.0,           // β
            &mut row_sums, // y
            1,             // incy
        );
    }

    // b) Scale each row by 1 / sumᵣ (Level-1 BLAS: SSCAL)
    for r in 0..m {
        let inv_sum = 1.0 / row_sums[r];
        unsafe {
            sscal(n as i32, inv_sum, &mut out[r * n..], 1);
        }
    }

    out
}

/// Power-iteration to estimate the dominant (left) eigenvector of an **n × n**
/// transition matrix.
///
/// * `transition` – flat row-major slice (length `n * n`)
/// * `increase_power` – square the matrix at every step (same semantics as the
///   original `ndarray` version)
/// * `max_iter` – iteration cap
///
/// Returns a `Vec<f32>` of length `n` holding the eigenvector.
///
/// Notes
/// -----
/// * The original code used `transition_matrix.t()` and then `dot`.  
///   Here we build that transpose once (`trans_mat`) and treat it as the
///   matrix we multiply by at every iteration (`trans_mat · v`).
/// * Matrix–vector products use **SGEMV** (Level-2 BLAS).  
///   Optional squaring uses **SGEMM** (Level-3 BLAS).
pub fn blas_power_method(
    transition: &[f32],
    n: usize,
    increase_power: bool,
    max_iter: usize,
) -> Vec<f32> {
    assert_eq!(transition.len(), n * n, "matrix must be n × n");

    // --- 0) Edge-case: 1×1 matrix -------------------------------------------
    if n == 1 {
        return vec![1.0];
    }

    // --- 1) Build Aᵀ because the ndarray version did transition_matrix.t() ----
    let mut trans_mat = vec![0.0_f32; n * n];
    for i in 0..n {
        // row in Aᵀ
        for j in 0..n {
            // col in Aᵀ
            trans_mat[i * n + j] = transition[j * n + i];
        }
    }

    // --- 2) Start with all-ones eigenvector ----------------------------------
    let mut v = vec![1.0_f32; n];
    let mut v_next = vec![0.0_f32; n];

    // Pre-allocate work buffer for SGEMV (the “x” input can be reused each time)
    // SGEMV will overwrite `v_next`.
    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    for _ in 0..max_iter {
        // a) v_next = trans_mat · v   (Level-2 BLAS)
        unsafe {
            sgemv(
                Layout::RowMajor,
                Transpose::None, // already Aᵀ
                n as i32,        // rows
                n as i32,        // cols
                alpha,
                &trans_mat,
                n as i32, // lda
                &v,       // x
                1,        // incx
                beta,
                &mut v_next, // y
                1,           // incy
            );
        }

        // b) Convergence test ‖v_next − v‖₂
        let mut diff_sq = 0.0_f32;
        for (a, b) in v_next.iter().zip(&v) {
            let d = a - b;
            diff_sq += d * d;
        }
        if diff_sq.sqrt() < 1e-5 {
            return v_next;
        }

        // c) Prepare for next iteration
        v.clone_from_slice(&v_next);

        // d) Optional power doubling:  trans_mat ← trans_mat · trans_mat
        if increase_power {
            let mut squared = vec![0.0_f32; n * n];
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    n as i32,     // m
                    n as i32,     // n
                    n as i32,     // k
                    1.0,          // α
                    &trans_mat,   // A
                    n as i32,     // lda
                    &trans_mat,   // B
                    n as i32,     // ldb
                    0.0,          // β
                    &mut squared, // C
                    n as i32,     // ldc
                );
            }
            trans_mat = squared; // replace with squared matrix
        }
    }

    v_next
}

/// Estimate the stationary distribution of an *n × n* transition matrix.
///
/// * `transition`  – flat row-major slice (`n * n` elements)  
/// * `increase_power`, `max_iter` – forwarded to `power_method`  
/// * `normalized`  – if `true`, divide the resulting vector by **n**
///
/// Returns a `Vec<f32>` of length *n*.
pub fn blas_stationary_distribution(
    transition: &[f32],
    n: usize,
    increase_power: bool,
    max_iter: usize,
    normalized: bool,
) -> Vec<f32> {
    // ---- 1) Sanity check ----------------------------------------------------
    assert_eq!(
        transition.len(),
        n * n,
        "Transition matrix must be square (n × n)"
    );

    // ---- 2) Dominant eigenvector via BLAS-powered power-iteration ----------
    let mut dist = blas_power_method(transition, n, increase_power, max_iter);

    // ---- 3) Optional “normalisation” (divide by n) --------------------------
    if normalized {
        let scale = 1.0_f32 / n as f32;
        unsafe {
            sscal(
                n as i32,  // number of elements
                scale,     // α
                &mut dist, // y
                1,         // stride
            );
        }
    }

    dist
}

#[inline(always)]
pub fn blas_norm2_f32(vec: &[f32]) -> f32 {
    unsafe { snrm2(vec.len() as i32, vec, 1) }
}

/// Above this many sentences, one `ssyrk` call beats the per-pair `sdot` loop.
///
/// Both compute one triangle of `E·Eᵀ`, the loop in `r(r-1)/2` short calls and `ssyrk` in one
/// blocked call. The loop's per-call cost dominates as `r` grows: at 768 dimensions and one
/// thread the two cross between 16 and 32 rows on every backend measured (the `similarity/`
/// bench group), and by 256 rows the loop is 3-18x slower. Below the threshold the loop still
/// wins, most clearly with Accelerate on the tall, skinny matrices LexRank gets from one chunk
/// (9-23 sentences x 384-1536 dimensions), where its `ssyrk` is about 2x slower than `sgemm`.
pub const SYRK_MIN_ROWS: usize = 32;

/// The cosine similarity matrix, by whichever route suits the shape: see [`SYRK_MIN_ROWS`].
pub fn blas_cosine_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    if r >= SYRK_MIN_ROWS {
        blas_cosine_f32_matrix_syrk(matrix, r, c)
    } else {
        blas_cosine_f32_matrix(matrix, r, c)
    }
}

/// The cosine similarity matrix through one `ssyrk` call.
///
/// `ssyrk` fills one triangle of the Gram matrix `E·Eᵀ`, whose diagonal holds `‖row‖²`, so the
/// norms come out of the same call. The triangle is then scaled by them and mirrored.
pub fn blas_cosine_f32_matrix_syrk(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c, "dimension mismatch");
    let mut sim = vec![0.0f32; r * r];
    unsafe {
        ssyrk(
            Layout::RowMajor,
            Part::Upper,
            Transpose::None,
            r as i32,
            c as i32,
            1.0,
            matrix,
            c as i32,
            0.0,
            &mut sim,
            r as i32,
        )
    };

    // `ssyrk` left ‖row i‖² on the diagonal.
    let norms: Vec<f32> = (0..r).map(|i| sim[i * r + i].sqrt()).collect();
    let inv_norm: Vec<f32> = norms
        .iter()
        .map(|norm| 1.0 / norm.max(f32::MIN_POSITIVE))
        .collect();
    for i in 0..r {
        sim[i * r + i] = 1.0;
        for j in (i + 1)..r {
            let s = sim[i * r + j] * inv_norm[i] * inv_norm[j];
            sim[i * r + j] = s;
            sim[j * r + i] = s;
        }
    }
    zero_rows_without_direction(&mut sim, &norms, r);
    sim
}

/// `‖x‖₂` as `sqrt(x · x)`.
///
/// `snrm2` scales as it accumulates so that it cannot overflow, and some libraries pay a lot for
/// that: on the DGX Spark, OpenBLAS's ARM `snrm2` took 14.8 µs of the 27.1 µs similarity step at
/// 1536 dimensions, against 2.5 µs for MKL. Embedding norms are far from `f32`'s limits, so the
/// plain dot product is safe here and 3-6x faster on that machine.
#[inline(always)]
pub fn blas_norm2_sdot_f32(vec: &[f32]) -> f32 {
    let n = vec.len() as i32;
    unsafe { sdot(n, vec, 1, vec, 1) }.sqrt()
}

#[inline(always)]
pub fn blas_cosine_f32_opt(a: &[f32], b: &[f32], a_norm: f32, b_norm: f32) -> f32 {
    //assert_eq!(a.len(), b.len(), "dimension mismatch");
    let dot = unsafe { sdot(a.len() as i32, a, 1, b, 1) };
    dot / (a_norm * b_norm)
}

pub fn blas_cosine_f32_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    //assert_eq!(matrix.len(), r * c);

    // ❶ allocate the square result (row-major)
    let mut result = vec![0.0f32; r * r];

    let mut norms = vec![0.0f32; r];
    norms.iter_mut().enumerate().for_each(|(i, norm)| {
        // -- row i norm --
        let row_i = &matrix[i * c..(i + 1) * c];
        *norm = blas_norm2_sdot_f32(row_i); // compute row i norm
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
                row_i[j] = blas_cosine_f32_opt(a, b, norms[i], norms[j]); // upper-tri entry
            }
        });

    // ❸ mirror the upper triangle → lower triangle (cheap, serial)
    for i in 0..r {
        for j in (i + 1)..r {
            let sim = result[i * r + j];
            result[j * r + i] = sim;
        }
    }

    // ❹ rows without a direction, whose `blas_cosine_f32_opt` came out as 0/0 = NaN
    zero_rows_without_direction(&mut result, &norms, r);

    result
}

pub fn blas_degree_centrality_scores(
    sim: &[f32],
    n: usize,
    increase_power: bool,
    threshold: Option<f32>,
    max_iter: usize,
    normalized: bool,
) -> anyhow::Result<Vec<f32>> {
    // ---- 1) Validate threshold ---------------------------------------------
    if let Some(t) = threshold {
        if !(0.0..1.0).contains(&t) {
            bail!("'threshold' must be in the interval [0, 1) or None");
        }
    }

    Ok(blas_centrality_scores(
        sim,
        n,
        increase_power,
        threshold,
        max_iter,
        normalized,
    ))
}

/// `blas_degree_centrality_scores` without the `[0, 1)` check: here `threshold` is an absolute
/// cosine similarity, which is negative whenever the embeddings have negative similarities.
fn blas_centrality_scores(
    sim: &[f32],
    n: usize,
    increase_power: bool,
    threshold: Option<f32>,
    max_iter: usize,
    normalized: bool,
) -> Vec<f32> {
    // ---- 2) Build the (row-stochastic) Markov matrix ------------------------
    let markov = if let Some(t) = threshold {
        blas_create_markov_matrix_discrete(sim, n, n, t)
    } else {
        blas_create_markov_matrix(sim, n, n)
    };

    // ---- 3) Stationary distribution (dominant left eigenvector) ------------
    blas_stationary_distribution(&markov, n, increase_power, max_iter, normalized)
}

/// LexRank scores for a batch of sentence (or token) embeddings.
///
/// * `embeddings` – flat slice of length `no_seq * embed_dim`  
/// * `no_seq`     – number of sequences (= rows)  
/// * `embed_dim`  – embedding dimensionality (= cols)  
/// * `threshold`  – if `Some(t)` ∈ [0,1), apply LexRank’s similarity thresholding trick  
/// * `max_iter`   – passed to the power-iteration inside the Markov chain
///
/// Returns a vector of `(index, score)` sorted from most to least central.
pub fn blas_lexrank_array(
    embeddings: &[f32],
    no_seq: usize,
    embed_dim: usize,
    threshold: Option<f32>,
    max_iter: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    // ------------------------------------------------------------------ early exit
    if embeddings.is_empty() {
        return Ok(vec![]);
    }
    if let Some(t) = threshold {
        if !(0.0..1.0).contains(&t) {
            bail!(
                "'threshold' must be in the interval [0, 1) or None, not {}",
                t
            );
        }
    }

    // ------------------------------------------------------------------ 1) similarity matrix
    // Row-major, length = no_seq²
    let sim_flat = blas_cosine_matrix(embeddings, no_seq, embed_dim);

    // ------------------------------------------------------------------ 2) adapt threshold to similarity range
    // `t` is relative (0 = least similar pair, 1 = identical); the absolute cut-off lies in
    // [sim_min, 1) and is negative when the embeddings have negative similarities, so it must
    // not go through blas_degree_centrality_scores' [0, 1) check.
    let thresh_adj = threshold.map(|t| {
        let sim_min = sim_flat.iter().copied().fold(f32::INFINITY, f32::min);
        let sim_range = 1.0 - sim_min; // max possible – min
        sim_min + t * sim_range // same formula as original
    });

    // ------------------------------------------------------------------ 3) degree-centrality (LexRank) scores
    let scores = blas_centrality_scores(
        &sim_flat, // similarity matrix (flat)
        no_seq,    // n × n
        /*increase_power=*/ false, thresh_adj, max_iter, /*normalized=*/ true,
    );

    // ------------------------------------------------------------------ 4) zip with indices and sort descending
    let mut ranked: Vec<(usize, f32)> = scores
        .into_iter()
        .enumerate() // (idx, score)
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    Ok(ranked)
}

pub fn blas_cosine_f32_matrix_opt(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c);

    // 1) G = M · Mᵀ   (r×c) · (c×r) -> (r×r)
    let mut gram = vec![0.0f32; r * r];
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,     // M
            Transpose::Ordinary, // Mᵀ
            r as i32,
            r as i32,
            c as i32,
            1.0,
            matrix,
            c as i32,
            matrix,
            c as i32,
            0.0,
            &mut gram,
            r as i32,
        );
    }

    // 2) norms (avoid div-by-zero)
    let norms: Vec<f32> = (0..r)
        .map(|i| blas_norm2_f32(&matrix[i * c..(i + 1) * c]))
        .collect();
    let scale: Vec<f32> = norms
        .iter()
        .map(|n| 1.0 / n.max(f32::MIN_POSITIVE))
        .collect();

    // 3) divide by outer product of norms: C[i,j] /= norms[i]*norms[j]
    //    First scale rows, then columns (sscal supports strided columns).
    for i in 0..r {
        unsafe { sscal(r as i32, scale[i], &mut gram[i * r..], 1) };
    }
    for j in 0..r {
        unsafe { sscal(r as i32, scale[j], &mut gram[j..], r as i32) };
    }

    // 4) clamp diagonal to 1 (small num errors)
    for i in 0..r {
        gram[i * r + i] = 1.0;
    }
    zero_rows_without_direction(&mut gram, &norms, r);
    gram
}

pub fn blas_softmax_opt(weights: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(weights.len(), m * n);
    let mut out = vec![0.0f32; m * n];

    // Option A: outer parallelism → set BLAS to single-thread (see §4).
    // rayon::scope(|s| { for r in 0..m { s.spawn(move |_| { ... }) } });
    out.chunks_mut(n).enumerate().for_each(|(r, dst)| {
        let row = &weights[r * n..(r + 1) * n];

        // 1) row max
        let xmax = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // 2) exp(x - xmax) and sum
        let mut sum = 0.0f32;
        for (d, &x) in dst.iter_mut().zip(row) {
            let e = (x - xmax).exp();
            *d = e;
            sum += e;
        }

        // 3) scale row by 1/sum (BLAS Level-1)
        let inv = 1.0 / sum;
        unsafe { sscal(n as i32, inv, dst, 1) };
    });

    out
}
