use ndarray::{Array, Array1, Array2, Axis, Ix0};
use rayon::prelude::*;

pub mod cblas_impl;
#[cfg(feature = "testing")]
pub mod testing;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
extern crate cblas;
#[cfg(any(feature = "blas", feature = "blas-static"))]
extern crate openblas_src;

use simsimd::SpatialSimilarity;

use anyhow::Result;
use std::ops::{MulAssign, Sub};

pub fn norm(tensor: &Array1<f32>) -> anyhow::Result<Array<f32, Ix0>> {
    Ok(tensor.pow2().sum_axis(Axis(0)).sqrt())
}

#[inline(always)]
pub fn normalize_l2(normed: &mut Array2<f32>) {
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

/// Whether a row's norm gives it a direction to compare against: `false` for an all-zero row,
/// and for one holding an infinity or a NaN.
///
/// The cosine of such a row is undefined. Computing it anyway divides by zero, and the NaN that
/// produces spreads through the Markov matrix into every score. `normalize_l2` substitutes a norm
/// of 1 and so leaves a zero row at zero, which makes the ndarray path return zeros for its whole
/// row and column, diagonal included; the cosine routines that do not go through `normalize_l2`
/// check this instead and write the same zeros.
#[inline(always)]
pub(crate) fn has_direction(norm: f32) -> bool {
    norm > 0.0 && norm.is_finite()
}

/// Zero the row and the column of every row that has no direction, diagonal included.
///
/// The cosine routines that divide by the norms leave `0/0 = NaN` in exactly those places, so
/// this runs once over the finished `r × r` matrix rather than testing every pair inside the
/// loops: skipping a row costs one comparison, and no embedding set in practice has such a row.
pub(crate) fn zero_rows_without_direction(sim: &mut [f32], norms: &[f32], r: usize) {
    for i in 0..r {
        if has_direction(norms[i]) {
            continue;
        }
        for j in 0..r {
            sim[i * r + j] = 0.0;
            sim[j * r + i] = 0.0;
        }
    }
}

#[inline(always)]
pub fn similarity_matrix(embeddings: &mut Array2<f32>) -> Array2<f32> {
    normalize_l2(embeddings);
    embeddings.dot(&embeddings.t())
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
    centrality_scores(
        similarity_matrix,
        increase_power,
        threshold,
        max_iter,
        normalized,
    )
}

/// `degree_centrality_scores` without the `[0, 1)` check: here `threshold` is an absolute
/// cosine similarity, which is negative whenever the embeddings have negative similarities.
fn centrality_scores(
    similarity_matrix: &Array2<f32>,
    increase_power: bool,
    threshold: Option<f32>,
    max_iter: usize,
    normalized: bool,
) -> Result<Array1<f32>> {
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
    lexrank_array(
        &embeddings_flatten,
        embeddings.len(),
        embeddings[0].len(),
        threshold,
        max_iter,
    )
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
    if let Some(threshold) = threshold {
        if !(0.0..1.0).contains(&threshold) {
            return Err(anyhow::anyhow!(
                "'threshold' must be in the interval [0, 1) or None, not {}",
                threshold
            ));
        }
    }
    let mut embeddings = Array2::from_shape_vec((no_seq, embed_dim), embeddings.clone())?;
    let sim_matrix = similarity_matrix(&mut embeddings);
    // `threshold` is relative: 0 is the least similar pair, 1 is identical. The absolute
    // cut-off lies in [sim_min, 1) and is negative when the embeddings have negative
    // similarities, so it must not go through degree_centrality_scores' [0, 1) check.
    let threshold = threshold.map(|threshold| {
        let sim_min: f32 = sim_matrix.flatten().into_iter().reduce(f32::min).unwrap();
        let sim_range = 1f32 - sim_min;
        sim_min + threshold * sim_range
    });
    let scores = centrality_scores(&sim_matrix, false, threshold, max_iter, true)?;
    let scores_vec: Vec<f32> = scores.flatten().to_vec();
    let mut ranked_sentences: Vec<_> = (0..no_seq as usize).zip(scores_vec).collect();
    ranked_sentences.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(ranked_sentences)
}

/// The cosine **similarity** matrix through SimSIMD, for comparison with the BLAS routes.
///
/// `simsimd`'s `f32::cosine` returns the cosine *distance*, `1 - similarity`, so each entry is
/// converted back. Rows without a direction are handled as `has_direction` describes, since
/// simsimd calls two zero rows identical (distance 0) and a zero row orthogonal to every other
/// (distance 1), neither of which matches the ndarray path.
pub fn ss_cosine_f32_matrix(matrix: &[f32], r: usize, c: usize) -> Vec<f32> {
    assert_eq!(matrix.len(), r * c);

    // ❶ allocate the square result (row-major)
    let mut result = vec![0.0f32; r * r];

    let norms: Vec<f32> = (0..r)
        .map(|i| {
            matrix[i * c..(i + 1) * c]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .collect();

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
                // `cosine` gives the distance; the matrix holds similarities.
                row_i[j] = 1.0 - f32::cosine(a, b).unwrap() as f32; // upper-tri entry
            }
        });

    // ❸ mirror the upper triangle → lower triangle (cheap, serial)
    for i in 0..r {
        for j in (i + 1)..r {
            let sim = result[i * r + j];
            result[j * r + i] = sim;
        }
    }

    // ❹ rows without a direction, which simsimd would otherwise call identical to each other
    zero_rows_without_direction(&mut result, &norms, r);

    result
}
