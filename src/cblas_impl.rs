
use cblas::*;
use rayon::prelude::*;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
extern crate cblas;
#[cfg(feature = "blas")]
extern crate openblas_src;

#[inline(always)]
pub fn blas_norm2_f32(vec: &[f32]) -> f32 {
    unsafe { snrm2(vec.len() as i32, vec, 1) }
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
        *norm = blas_norm2_f32(row_i); // compute row i norm
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

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use simsimd::SpatialSimilarity;

    fn rand_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut embeds = vec![0f32; rows * cols];
        embeds.par_chunks_mut(cols).for_each(|chunk| {
            for i in 0..cols {
                chunk[i] = rand::random::<f32>();
            }
        });
        embeds
    }
    #[test]
    fn test_cosine_f32_matrix() {
        let rows = 3;
        let cols = 4;
        let embeds = rand_matrix(rows, cols);
        let result = blas_cosine_f32_matrix(&embeds, rows, cols);
        assert_eq!(result.len(), rows * rows);
        let sim_array = Array2::from_shape_vec((rows, rows), result).unwrap();
        println!("{:8.16}", sim_array);

        let mut simsimd_result = vec![0.0f32; rows * rows];
        for i in 0..rows {
            let a = &embeds[i * cols..(i + 1) * cols];
            for j in 0..rows {
                let b = &embeds[j * cols..(j + 1) * cols];
                let ss_cosine = 1.0 - f32::cosine(a, b).unwrap();
                simsimd_result[i * rows + j] = ss_cosine as f32;
                simsimd_result[j * rows + i] = ss_cosine as f32; // mirror
            }
        }
        let simsimd_array = Array2::from_shape_vec((rows, rows), simsimd_result).unwrap();
        println!("{:8.16}", simsimd_array);
    }
}
