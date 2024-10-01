#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::ops::{Div, Mul};
use ndarray::{s, Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

pub fn get_rand_arr2_f32(
    m: usize,
    n: usize,
    mean: f32,
    std_dev: f32,
) -> anyhow::Result<Array2<f32>> {
    let rnd_arr = Array::random((m, n), Normal::new(mean, std_dev)?).into();
    Ok(rnd_arr)
}

pub fn get_rand_arr1_f32(n: usize, mean: f32, std_dev: f32) -> anyhow::Result<Array1<f32>> {
    let rnd_arr = Array::random(n, Normal::new(mean, std_dev)?).into();
    Ok(rnd_arr)
}

pub fn linear_forward(
    lhs: &Array2<f32>,
    rhs: &Array2<f32>,
    bias: &Array1<f32>,
) -> anyhow::Result<Array2<f32>> {
    let mul = lhs.dot(&rhs.t());
    let add = mul + bias;
    Ok(add)
}

pub fn normalize_l2(embeddings: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let norm = embeddings.pow2().sum_axis(Axis(1)).sqrt().clamp(1e-12, f32::MAX);
    let normed = embeddings / norm.insert_axis(Axis(1));
    Ok(normed)
}

pub fn cosine_sim(
    embeddings: &Array2<f32>,
) -> anyhow::Result<Array2<f32>> {
    let normed = normalize_l2(embeddings)?;
    let sim_matrix = normed.dot(&normed.t());
    Ok(sim_matrix)
}
