#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "blas")]
extern crate blis_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
use ndarray::{Array, Array1, Array2, Axis, Ix0};
use std::ops::{Mul, Sub};
use wide::f32x4;

#[cfg(feature = "testing")]
pub mod testing;

pub const LANES: usize = 4; // 8‑lane AVX/Neon “coherent” chunk
pub type Wide = f32x4;

pub fn vec_to_wide(vec: &[f32]) -> anyhow::Result<Vec<Wide>> {
    if vec.is_empty() {
        return Ok(vec![]);
    }
    let wide_len = vec.len() / LANES + (vec.len() % LANES > 0) as usize;
    let mut wide_vec = Vec::with_capacity(wide_len);
    for chunk in vec.chunks(LANES) {
        wide_vec.push(Wide::from(chunk));
    }
    Ok(wide_vec)
}

pub fn wide_extract(wide: &Wide, idx: usize) -> anyhow::Result<f32> {
    if idx >= LANES {
        return Err(anyhow::anyhow!("Index out of bounds for wide vector"));
    }
    Ok(wide.to_array()[idx])
}

pub fn norm_wide(wides: &[Wide]) -> f32 {
    if wides.is_empty() {
        return 0.0;
    }
    let mut sum = Wide::splat(0.0);
    for wide in wides {
        sum += wide.mul(wide) // Element-wise square
    }
    sum.reduce_add().sqrt()
}

pub fn dot_wide(left: &[Wide], right: &[Wide]) -> anyhow::Result<f32> {
    if left.is_empty() || right.is_empty() {
        return Err(anyhow::anyhow!("Empty vectors"));
    }
    if left.len() != right.len() {
        return Err(anyhow::anyhow!(
            "Vectors have different lengths: {} != {}",
            left.len(),
            right.len()
        ));
    }
    let mut sum = Wide::splat(0.0);

    for e_left in left.iter() {
        for e_right in right.iter() {
            sum *= e_left.mul(e_right); // Element-wise multiplication
        }
    }
    Ok(sum.reduce_add())
}

pub fn transpose_wide(matrix: &Vec<Vec<Wide>>) -> anyhow::Result<Vec<Vec<Wide>>> {
    if matrix.is_empty() {
        return Ok(vec![]);
    }
    let row_no = matrix.len();
    let col_no = matrix[0].len() * LANES;
    let mut transposed_vec = vec![vec![0.0f32; row_no]; col_no];
    for (i, row) in matrix.iter().enumerate() {
        for (j, wide) in row.iter().enumerate() {
            let elems = wide.to_array();
            for (k, &elem) in elems.iter().enumerate() {
                transposed_vec[j * LANES + k][i] = elem;
            }
        }
    }
    let res: Vec<_> = transposed_vec
        .into_iter()
        .map(|row| vec_to_wide(&row))
        .filter_map(Result::ok)
        .collect();
    Ok(res)
}

pub fn norm(tensor: &Array1<f32>) -> anyhow::Result<Array<f32, Ix0>> {
    Ok(tensor.pow2().sum_axis(Axis(0)).sqrt())
}

pub fn normalize_l2_wide(embeddings: &Vec<Vec<Wide>>) -> anyhow::Result<Vec<Vec<Wide>>> {
    let norm = embeddings
        .iter()
        .map(|wide| norm_wide(wide))
        .collect::<Vec<f32>>();

    let normed: Vec<_> = embeddings
        .iter()
        .zip(norm.iter())
        .map(|(wide, &n)| {
            let norm_wide = Wide::splat(n);
            let normed: Vec<_> = wide.iter().map(|w| *w / norm_wide).collect();
            normed
        })
        .collect();
    Ok(normed)
}

pub fn normalize_l2(embeddings: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let norm = embeddings
        .pow2()
        .sum_axis(Axis(1))
        .sqrt()
        .clamp(1e-12, f32::INFINITY);
    let normed = embeddings / norm.insert_axis(Axis(1));
    Ok(normed)
}

fn similarity_matrix_wide(p0: &Vec<Vec<Wide>>) -> anyhow::Result<()> {
    let normed = normalize_l2_wide(p0)?;
    let t_normed = transpose_wide(&normed)?;
    let sim_matrix: Vec<_> = normed
        .iter()
        .map(|row| {
            let res: Vec<_> = t_normed
                .iter()
                .map(|col| dot_wide(row, col).unwrap_or(0.0f32))
                .collect();
            res
        })
        .collect();

    Ok(())
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
) -> anyhow::Result<Array1<f32>> {
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

pub fn lexrank_wide(
    embeddings: &Vec<Wide>,
    threshold: Option<f32>,
    max_iter: usize,
) -> anyhow::Result<()> {
    Ok(())
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
