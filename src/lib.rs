use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use ndarray::parallel::prelude::*;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;



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
    let mul = lhs.dot(rhs);
    let add = mul + bias;
    Ok(add)
}
