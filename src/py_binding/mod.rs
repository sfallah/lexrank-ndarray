use pyo3::prelude::*;
use pyo3::exceptions;
use ndarray::Array2;
//use crate::lexrank_ndarray::{lexrank_ts, normalize_l2, similarity_matrix};
use crate::lexrank_ts;

#[pyfunction]
fn lexrank_py(
    _py: Python,
    embeds: Vec<Vec<f32>>,  
    max_iter: usize,
    threshold: Option<f32>,

) -> PyResult<Vec<(usize, f32)>> {

    // Validate that all inner vectors have the same length
    if !embeds.is_empty() {
        let cols = embeds[0].len();
        if embeds.iter().any(|row| row.len() != cols) {
            return Err(exceptions::PyValueError::new_err("All rows must have the same number of columns."));
        }
    }

    // Convert Vec<Vec<f32>> to Array2<f32>
    let rows = embeds.len();
    let cols = if rows > 0 { embeds[0].len() } else { 0 };
    let flat_data: Vec<f32> = embeds.into_iter().flatten().collect();

    let embeds_array = Array2::from_shape_vec((rows, cols), flat_data)
        .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid shape: {}", e)))?;

    // Call the existing lexrank_ts function
    match lexrank_ts(&embeds_array, threshold, max_iter) {
        Ok(ranked_sentences) => Ok(ranked_sentences),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

#[pymodule]
fn lexrank_ndarray(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lexrank_py, m)?)?;
    // Add more functions as needed
    Ok(())
}