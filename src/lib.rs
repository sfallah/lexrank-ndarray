#![feature(portable_simd)]

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "blas")]
extern crate blis_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use ndarray::{Array, Array1, Array2, Axis, Ix0};
use std::ops::{MulAssign, Sub};

#[cfg(feature = "testing")]
pub mod testing;
mod wide_impl;

pub use wide_impl::*;

pub fn norm(tensor: &Array1<f32>) -> anyhow::Result<Array<f32, Ix0>> {
    Ok(tensor.pow2().sum_axis(Axis(0)).sqrt())
}

use rayon::prelude::*;

#[inline]
pub fn normalize_l2(embeddings: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let norm = embeddings
        .pow2()
        .sum_axis(Axis(1))
        .sqrt()
        .mapv(|x| if x > 0.0 { x } else { 1f32 })
        .recip();
    let normed = embeddings * norm.insert_axis(Axis(1));
    Ok(normed)
}

pub fn normalize_l2_par(normed: &mut Array2<f32>) {
    normed
        .axis_iter_mut(Axis(0))
        .into_iter()
        .for_each(|mut row| {
            let mut norm = row.pow2().sum().sqrt();
            if norm <= 0.0 {
                norm = 1.0; // Avoid division by zero
            }
            row.mul_assign(norm.recip());
        });
}

#[inline]
pub fn similarity_matrix_par(embeddings: &mut Array2<f32>) {
    // Normalize in-place
    normalize_l2_par(embeddings);

    // Compute cosine similarity as dot product of normalized vectors
    embeddings.dot(&embeddings.t());
}

pub fn similarity_matrix(embeddings: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let normed = normalize_l2(embeddings)?;
    let sim_matrix = normed.dot(&normed.t());
    Ok(sim_matrix)
}

pub fn cos_similarity(embedding1: &Vec<f32>, embedding2: &Vec<f32>) -> anyhow::Result<f32> {
    if embedding1.is_empty() || embedding2.is_empty() {
        return Err(anyhow::anyhow!("Empty embeddings"));
    }
    let embed1_shape = embedding1.len();
    let embed2_shape = embedding2.len();
    if embed1_shape != embed2_shape {
        return Err(anyhow::anyhow!(
            "Embeddings have different shapes: {} != {}",
            embed1_shape,
            embed2_shape
        ));
    }
    let array1: Array1<f32> = Array::from(embedding1.to_vec());
    let array2: Array1<f32> = Array::from(embedding2.to_vec());
    let normed1 = array1.clone() / norm(&array1)?.clamp(1e-12, f32::INFINITY);
    let normed2 = array2.clone() / norm(&array2)?.clamp(1e-12, f32::INFINITY);
    let sim = normed1.dot(&normed2.t());
    Ok(sim)
}

/// Threshold each coefficient (`>= threshold → 1.0, else 0.0`)
/// then scale every row so it sums to 1 (stochastic/Markov form).
///
/// Returns an error if a row’s sum is 0 or non‑finite.

pub fn create_markov_matrix_discrete(
    weights_matrix: &Array2<f32>,
    threshold: f32,
) -> anyhow::Result<Array2<f32>> {
    let discrete_weights_matrix =
        weights_matrix.mapv(|x| if x >= threshold { 1.0f32 } else { 0.0f32 });
    create_markov_matrix(&discrete_weights_matrix)
}
pub fn create_markov_matrix(weights_matrix: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let min = weights_matrix
        .flatten()
        .into_iter()
        .reduce(f32::min)
        .unwrap();
    if min <= 0.0 {
        Ok(softmax(weights_matrix)?)
    } else {
        let row_sum = weights_matrix.sum_axis(Axis(1));
        //println!("create_markov_matrix row_sum {:?}", row_sum.shape());
        Ok(weights_matrix / row_sum.insert_axis(Axis(1)))
    }
}

pub fn softmax(weights_matrix: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let exp_vals = weights_matrix.mapv(f32::exp);
    let exp_sum = exp_vals.sum_axis(Axis(1));
    Ok(exp_vals / exp_sum.insert_axis(Axis(1)))
}

/// Numerically‑stable softmax over each row.
///
/// * If the matrix is empty the result is empty.
/// * Every row keeps the same SIMD chunking as the input (no re‑packing).

pub fn degree_centrality_scores(
    similarity_matrix: &Array2<f32>,
    increase_power: bool,
    threshold: Option<f32>,
    max_iter: usize,
    normalized: bool,
) -> Result<Array1<f32>> {
    if threshold.is_some() {
        let threshold = threshold.unwrap();
        assert!(
            threshold >= 0.0 && threshold < 1.0,
            "'threshold' should be a floating-point number from the interval [0, 1) or None"
        );
    }
    let markov_matrix = if let Some(threshold) = threshold {
        create_markov_matrix_discrete(&similarity_matrix, threshold)?
    } else {
        create_markov_matrix(similarity_matrix)?
    };

    let scores = stationary_distribution(&markov_matrix, increase_power, max_iter, normalized)?;

    Ok(scores)
}

pub fn stationary_distribution(
    transition_matrix: &Array2<f32>,
    increase_power: bool,
    max_iter: usize,
    normalized: bool,
) -> anyhow::Result<Array1<f32>> {
    let tr_mx_dims = transition_matrix.shape();
    assert_eq!(
        tr_mx_dims[0], tr_mx_dims[1],
        "Transition matrix should be square"
    );

    let mut distribution = power_method(&transition_matrix, increase_power, max_iter)?;

    if normalized {
        distribution = distribution / tr_mx_dims[0] as f32;
    }

    Ok(distribution)
}

pub fn power_method(
    transition_matrix: &Array2<f32>,
    increase_power: bool,
    max_iter: usize,
) -> anyhow::Result<Array1<f32>> {
    let n = transition_matrix.shape()[0];

    let mut eigenvector = Array::ones(n);

    if n == 1 {
        return Ok(eigenvector);
    }

    let mut transition: Array2<f32> = transition_matrix.t().to_owned();

    for _idx in 0..max_iter {
        let eigenvector_next = transition.dot(&eigenvector);

        let lm_val: f32 = norm(&eigenvector_next.clone().sub(&eigenvector))?.into_scalar();
        if lm_val < 1e-5 {
            return Ok(eigenvector_next);
        }
        eigenvector = eigenvector_next;

        if increase_power {
            transition = transition.clone().dot(&transition);
        }
    }

    Ok(eigenvector.into())
}

pub fn lexrank(
    embeddings: &Vec<Vec<f32>>,
    threshold: Option<f32>,
    max_iter: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    if embeddings.is_empty() {
        return Ok(vec![]);
    }
    let embeddings_flatten: Vec<f32> = embeddings.iter().flatten().cloned().collect();
    let embeddings_array: Array2<f32> = Array::from(embeddings_flatten)
        .into_shape_clone((embeddings.len(), embeddings[0].len()))?;
    lexrank_ts(&embeddings_array, threshold, max_iter)
}

pub fn lexrank_array(
    embeddings: &Vec<f32>,
    no_seq: usize,
    embed_dim: usize,
    threshold: Option<f32>,
    max_iter: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    if embeddings.is_empty() {
        return Ok(vec![]);
    }
    let embeddings_array: Array2<f32> =
        Array::from(embeddings.to_vec()).into_shape_clone((no_seq, embed_dim))?;
    lexrank_ts(&embeddings_array, threshold, max_iter)
}

pub fn lexrank_ts(
    embeddings_array: &Array2<f32>,
    threshold: Option<f32>,
    max_iter: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    if embeddings_array.shape()[0] == 0 {
        return Ok(vec![]);
    }
    let sim_matrix = similarity_matrix(&embeddings_array)?;
    let threshold = threshold.map(|threshold| {
        let sim_min: f32 = sim_matrix.flatten().into_iter().reduce(f32::min).unwrap();
        //println!("sim_min: {:8.16}", sim_min);
        let sim_range = 1f32 - sim_min;
        sim_min + threshold * sim_range
    });
    let scores = degree_centrality_scores(&sim_matrix, false, threshold, max_iter, true)?;
    let scores_vec: Vec<f32> = scores.flatten().to_vec();
    let mut ranked_sentences: Vec<_> = (0..embeddings_array.shape()[0] as usize)
        .zip(scores_vec)
        .collect();
    ranked_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(ranked_sentences)
}

/// Cosine-similarity matrix (row⋅row) with upper-triangle parallel fill.
///
/// * `embeddings.shape()` == (n, d)
/// * Requires `ndarray = { version = "0.15", features = ["rayon"] }`
#[inline]
pub fn similarity_matrix_par_new(embeddings: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    // 1.  Normalise rows in a single owned buffer
    let mut normed = embeddings.to_owned(); // one allocation
    normalize_l2_par(&mut normed); // in-place, parallel

    // 2.  Pre-allocate result (Row-major layout is ndarray’s default)
    let n = normed.nrows();
    let mut sim = Array2::<f32>::zeros((n, n));

    // 3.  Parallel fill: diagonal + upper triangle
    //
    //     `axis_iter_mut(Axis(0)).into_par_iter()` gives each Rayon
    //     worker a unique mutable row slice →   ✓ no aliasing
    //     We only write (i,i) and (i, j > i) inside that slice.
    //
    sim.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut sim_row)| {
            // Self-similarity
            sim_row[i] = 1.0;

            let row_i = normed.row(i);
            for j in (i + 1)..n {
                // SIMD - accelerated dot product from ndarray
                let val = row_i.dot(&normed.row(j));
                sim_row[j] = val; // upper half
            }
        });

    // 4.  Mirror upper → lower half (single thread, cache-friendly)
    for i in 0..n {
        for j in (i + 1)..n {
            sim[(j, i)] = sim[(i, j)];
        }
    }

    Ok(sim)
}

#[cfg(feature = "accelerate")]
use cblas::{
    sdot,
    sgemm,
    snrm2, // C-level calls exposed safely
    Layout,
    Transpose, // enum wrappers
};

/// Single-precision cosine similarity of two equal-length vectors.
#[cfg(feature = "accelerate")]
pub fn cosine_f32_old(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as i32;

    // BLAS level-1 already has both primitives we need:
    let dot = unsafe { sdot(n, a, 1, b, 1) };
    let na = unsafe { snrm2(n, a, 1) };
    let nb = unsafe { snrm2(n, b, 1) };

    dot / (na * nb)
}

pub fn cosine_f32_opt(a: &[f32], b: &[f32], a_norm: f32, b_norm: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut dot = 0.0f32;
    unsafe {
        vDSP_dotpr(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            &mut dot,
            a.len() as vDSP_Length,
        )
    };
    dot / (a_norm * b_norm)
}
/// # Parameters
/// * `matrix` – flat row-major buffer of size `r × c`
/// * `r` – number of rows (vectors)
/// * `c` – dimensionality of each vector
pub fn cosine_f32_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c);

    // ❶ allocate the square result (row-major)
    let mut result = vec![0.0f32; r * r];

    let mut norms = vec![0.0f32; r];
    norms.iter_mut().enumerate().for_each(|(i, norm)| {
        // -- row i norm --
        let row_i = &matrix[i * c..(i + 1) * c];
        *norm = norm2_f32(row_i); // compute row i norm
    });

    // ❷ process each *row slice* of `result` in parallel
    //
    // `par_chunks_mut(r)` splits the buffer into disjoint mutable chunks,
    // one per row, so every thread owns a unique region and no locks are needed.
    result
        .chunks_mut(r) // &mut [f32] for one row
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

mod ffi;
use ffi::*;
use std::slice;

#[inline]
fn norm2_f32(x: &[f32]) -> f32 {
    let mut ssq = 0.0_f32;
    unsafe { vDSP_svesq(x.as_ptr(), 1, &mut ssq, x.len() as vDSP_Length) };
    ssq.sqrt()
}

#[inline]
fn norm2_f64(x: &[f64]) -> f64 {
    let mut ssq = 0.0_f64;
    unsafe { vDSP_svesqD(x.as_ptr(), 1, &mut ssq, x.len() as vDSP_Length) };
    ssq.sqrt()
}

/// Single-precision cosine similarity of two equal-length slices.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut dot = 0.0f32;
    unsafe {
        vDSP_dotpr(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            &mut dot,
            a.len() as vDSP_Length,
        )
    };
    dot / (norm2_f32(a) * norm2_f32(b))
}

/// Double-precision cosine similarity.
pub fn cosine_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut dot = 0.0f64;
    unsafe {
        vDSP_dotprD(
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            &mut dot,
            a.len() as vDSP_Length,
        )
    };
    dot / (norm2_f64(a) * norm2_f64(b))
}



/// Build an m × n cosine-similarity matrix between two row-major matrices
/// stored as flat vectors (row stride = d).  The result is returned in C.
pub fn cosine_matrix_f32(
    a: &[f32],
    m: usize, // A: m × d
    b: &[f32],
    n: usize, // B: n × d
    d: usize, // shared dimension
) -> Vec<f32> {
    assert_eq!(a.len(), m * d);
    assert_eq!(b.len(), n * d);

    // 1) Dot-product matrix C = A · Bᵀ  (m × n)
    let mut c = vec![0f32; m * n];
    unsafe {
        // CBLAS uses enum ints; 101 = RowMajor, 111 = NoTrans
        cblas_sgemm(
            101,
            111,
            111,
            m as i32,
            n as i32,
            d as i32,
            1.0,
            a.as_ptr(),
            d as i32,
            b.as_ptr(),
            d as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }

    // 2) row and column ‖·‖₂ norms
    let mut row_norms = Vec::with_capacity(m);
    for i in 0..m {
        row_norms.push(norm2_f32(&a[i * d..(i + 1) * d]));
    }
    let mut col_norms = Vec::with_capacity(n);
    for j in 0..n {
        // take j-th row of Bᵀ i.e. j-th vector in B
        let col = (0..d).map(|k| b[j + k * n]).collect::<Vec<_>>();
        col_norms.push(norm2_f32(&col));
    }

    // 3) elementwise scale: C[i,j] /= (‖ai‖ · ‖bj‖)
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] /= row_norms[i] * col_norms[j];
        }
    }
    c
}
