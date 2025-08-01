use core::simd::f32x4;
use rayon::prelude::*;
use std::ops::{AddAssign, Div, DivAssign, MulAssign};
use std::simd::prelude::*;
use std::simd::StdFloat;
// SIMD vector type for 8 × f32
use anyhow::{anyhow, Result};
use ndarray::{Array, Array2};
use smallvec::SmallVec;
// SIMD lane = 4 × f32

// ────────────────────────────────────────────────────────────
// small type helpers – change the inline capacities to taste
pub type Wide = f32x4;
pub const LANES: usize = 4;

/// A row of SIMD values.
/// We keep `LANES` items inline; if we push more, SmallVec spills to the heap.
pub type WideRow = SmallVec<[Wide; LANES]>;

/// Keep four rows inline before spilling; pick a different number if you wish.
pub type WideMatrix = SmallVec<[WideRow; LANES]>;

// ────────────────────────────────────────────────────────────
// SIMD helpers                                                  */
#[inline(always)]
pub fn vec_to_row(v: &[f32]) -> WideRow {
    let mut out = SmallVec::with_capacity(wide_size(v.len()));
    for chunk in v.chunks(LANES) {
        out.push(Wide::load_or(chunk, Wide::splat(0.0)));
    }
    out
}

pub fn wide_matrix_to_array(matrix: &WideMatrix) -> Array2<f32> {
    let n_rows = matrix.len();
    if n_rows == 0 {
        return Array2::default((0, 0));
    }
    let n_cols = matrix[0].len() * LANES; // each WideRow has `LANES` f32 values
    let mut flat: Vec<f32> = Vec::with_capacity(n_rows * n_cols);
    for row in matrix {
        for wide in row {
            flat.extend_from_slice(&wide.to_array());
        }
    }
    Array::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

pub fn max_wide_matrix(matrix: &WideMatrix) -> Option<f32> {
    // 1. running SIMD maximum, one register wide
    let mut max_wide = f32::MIN;

    for row in matrix.iter() {
        //     ^ SmallVec<T> already derefs to a slice, so `.iter()` is cheap
        for &wide in row {
            // &Wide → Wide (Copy)
            let wide_max = wide.reduce_max(); // lane‑wise max :contentReference[oaicite:0]{index=0}
            max_wide = if wide_max > max_wide {
                wide_max
            } else {
                max_wide
            }; // lane‑wise max :contentReference[oaicite:0]{index=0}
        }
    }

    // 2. horizontal reduction to one scalar
    Some(max_wide)
}

pub fn min_wide_matrix(matrix: &WideMatrix) -> Option<f32> {
    // 1. running SIMD maximum, one register wide
    // 1. running SIMD maximum, one register wide
    let mut max_wide = f32::MIN;

    for row in matrix.iter() {
        //     ^ SmallVec<T> already derefs to a slice, so `.iter()` is cheap
        for &wide in row {
            // &Wide → Wide (Copy)
            let wide_max = wide.reduce_max(); // lane‑wise max :contentReference[oaicite:0]{index=0}
            max_wide = if wide_max > max_wide {
                wide_max
            } else {
                max_wide
            }; // lane‑wise max :contentReference[oaicite:0]{index=0}
        }
    }

    // 2. horizontal reduction to one scalar
    Some(max_wide)
}

#[inline(always)]
fn dot_wide_mm(a: &WideRow, b: &WideRow) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // Each Wide = 4 *f32*, so k = logical dimension
    let k = a.len() * LANES;

    // Treat the two WideRow slices as contiguous f32 buffers.
    // `wide::f32x4` is 16 bytes of four little-endian floats, laid out
    // sequentially in memory, so this `as *const f32` is sound.
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    // Result scalar C (1 × 1 matrix)
    let mut c = 0.0f32;

    unsafe {
        // C ← 1 · A · B  +  0 · C
        // A: (1 × k)   row-major,  row_stride = k, col_stride = 1
        // B: (k × 1)   row-major,  row_stride = 1, col_stride = 1
        // C: (1 × 1)   row-major,  row_stride = 1, col_stride = 1
        sgemm(
            1,   // m
            k,   // k
            1,   // n
            1.0, // alpha
            a_ptr,
            k as isize,
            1, // A strides
            b_ptr,
            1,
            1,   // B strides
            0.0, // beta
            &mut c as *mut f32,
            1,
            1, // C strides
        );
    }

    c
}

#[inline(always)]
pub fn flatten_vec_to_wide_matrix(vec: &[f32], n_rows: usize, n_cols: usize) -> Result<WideMatrix> {
    if vec.is_empty() || n_rows == 0 || n_cols == 0 {
        return Ok(SmallVec::new());
    }
    let mut wide_matrix: WideMatrix = SmallVec::with_capacity(n_rows);
    for row in vec.chunks(n_cols) {
        wide_matrix.push(vec_to_row(row));
    }
    Ok(wide_matrix)
}

pub fn vec_to_wide_matrix(vec: &[Vec<f32>]) -> Result<WideMatrix> {
    if vec.is_empty() {
        return Ok(SmallVec::new());
    }
    let n_rows = vec.len();
    let n_cols = vec[0].len();
    flatten_vec_to_wide_matrix(&vec.concat(), n_rows, n_cols)
}
#[inline(always)]
pub fn norm_wide(wides: &[Wide]) -> f32 {
    if wides.is_empty() {
        return 0.0;
    }
    let mut sum = Wide::splat(0.0);
    for wide in wides {
        sum += *wide * *wide; // Element-wise square
    }
    sum.reduce_sum().sqrt() // Horizontal sum and square root
}

/// Row‑wise L2 normalisation: each row is divided by its own ‖row‖₂.
///
/// Fails if a row’s norm is zero or non‑finite.
/// Row‑wise L2 normalisation: each row is divided by its own ‖row‖₂.
///
/// Fails if a row’s norm is zero or non‑finite.
pub fn normalize_l2_wide(matrix: &WideMatrix) -> Result<WideMatrix> {
    let mut out = SmallVec::with_capacity(matrix.len());

    for row in matrix {
        // ---------- 1. compute this row's squared‑L2 norm ----------
        let mut row_sum = Wide::splat(0.0); // accumulate the sum of squares
        for &chunk in row {
            row_sum = chunk.mul_add(chunk, row_sum); // four lanes at once
        }
        let row_norm = row_sum.reduce_sum().sqrt(); // horizontal sum and sqrt

        // ---------- 2. scale this row only ----------
        //let scale = Wide::splat(1.0 / row_norm); // multiply beats divide
        let mut normed_row: WideRow = SmallVec::with_capacity(row.len());
        let scale = Wide::splat(row_norm);

        for &chunk in row {
            normed_row.push(chunk.div(scale)); // divide each lane by the norm
        }

        out.push(normed_row);
    }

    Ok(out)
}

#[inline]
pub fn normalize_l2_wide_new(matrix: &mut WideMatrix) {
    matrix.iter_mut().for_each(|row| {
        // 1. SIMD accumulator for ∑x²
        let mut acc = Wide::splat(0.0);
        for &chunk in row.iter() {
            acc.add_assign( chunk * chunk); // accumulate the sum of squares
        }
        let scale = acc.reduce_sum().sqrt().clamp(1e-12, f32::INFINITY); // avoid division by zero
        let inv = Wide::splat(scale).recip(); // inverse norm

        // 2. scale
        for chunk in row.iter_mut() {
            chunk.mul_assign(inv); // multiply each lane by the inverse norm
        }
    })
}

/// Row‑wise L2 normalisation — now parallel with Rayon.
///
/// Each row’s work is completely independent, so the algorithm scales
/// almost linearly with the number of CPU cores.
///
/// *Fails* if any row has a zero or non‑finite norm.
pub fn normalize_l2_wide_par(matrix: &WideMatrix) -> Result<WideMatrix> {
    // 1. do every row on a different thread
    let rows: Vec<WideRow> = matrix
        .as_slice() // slice → `ParallelSlice`
        .par_iter()
        .map(|row| {
            // --- 1a. squared L2 norm (SIMD add) --------------------------
            let mut sum_sq = 0.0f32;
            for &chunk in row {
                sum_sq += (chunk * chunk).reduce_sum(); // four lanes at once
            }

            // --- 1b. scale the entire row --------------------------------
            let scale = Wide::splat(1.0 / sum_sq.sqrt());
            let mut normed: WideRow = SmallVec::with_capacity(row.len());
            for &chunk in row {
                normed.push(chunk * scale);
            }
            normed
        })
        .collect();

    Ok(SmallVec::from(rows))
}
pub fn normalize_l2_wide_old(embeddings: &WideMatrix) -> Result<WideMatrix> {
    let norm: SmallVec<[f32; 4]> = embeddings.iter().map(|wide| norm_wide(wide)).collect();

    let normed: WideMatrix = embeddings
        .iter()
        .zip(norm.iter())
        .map(|(wide, &n)| {
            let norm_wide = Wide::splat(n);
            let normed: WideRow = wide.iter().map(|w| *w / norm_wide).collect();
            normed
        })
        .collect();
    Ok(normed)
}

pub fn similarity_matrix_wide(p0: &WideMatrix) -> Result<Array2<f32>> {
    let normed = normalize_l2_wide(p0)?;
    let mut sims = Vec::with_capacity(p0.len() * p0.len());
    for row in normed.iter() {
        for col in normed.iter() {
            let sim = dot_wide(row, col);
            sims.push(sim);
        }
    }
    let sim_matrix: Array2<f32> = Array::from_shape_vec((p0.len(), p0.len()), sims)?;
    Ok(sim_matrix)
}

pub fn similarity_matrix_wide_par(p0: &WideMatrix) -> Result<Array2<f32>> {
    // 1. row‑wise L2 normalisation (unchanged, still SIMD)
    let normed = normalize_l2_wide(p0)?;
    let n = normed.len();

    // 2. compute all row–column dot products in parallel
    //
    //    normed.par_iter()  gives us a `ParallelIterator` over &WideRow.
    //    `flat_map_iter` expands each row into n dot products, so the final
    //    collect builds one flat Vec<f32> of length n*n.
    let sims: Vec<f32> = normed
        .par_iter()
        .flat_map_iter(|row| normed.iter().map(move |col| dot_wide(row, col)))
        .collect();

    // 3. reshape into the dense matrix expected by the caller
    Ok(Array2::from_shape_vec((n, n), sims)?)
}

pub fn similarity_matrix_wide_opt(p0: &WideMatrix) -> Result<WideMatrix> {
    let normed = normalize_l2_wide(p0)?;
    let n = normed.len();

    // 1. allocate the whole dense matrix once
    let mut sims = vec![0.0f32; n * n];

    // 2. fill the upper triangle (including the diagonal)
    for i in 0..n {
        // the diagonal is exactly 1 after L2 normalisation
        sims[i * n + i] = 1.0;

        for j in (i + 1)..n {
            let sim = dot_wide(&normed[i], &normed[j]);
            // row‑major indexing
            sims[i * n + j] = sim;
            sims[j * n + i] = sim; // mirror to the lower triangle
        }
    }

    Ok(flatten_vec_to_wide_matrix(&sims, n, n)?)
}

use matrixmultiply::sgemm; // crate = "matrixmultiply" in Cargo.toml

pub fn similarity_matrix_mm(p0: &WideMatrix) -> Result<WideMatrix> {
    // 0.  L2 normalise as before (still SIMD & optionally Rayon)
    let rows = p0.len();
    let cols = p0[0].len() * LANES; // SIMD lanes → scalar width
    let mut normed = p0.clone();
    normalize_l2_wide_new(&mut normed);

    // 1.  Flatten into one contiguous row‑major matrix A (rows × cols)
    let mut a = Vec::<f32>::with_capacity(rows * cols);
    for row in &normed {
        for chunk in row {
            // Wide → [f32; 4]
            a.extend_from_slice(&chunk.to_array());
        }
    }

    // 2.  Allocate C = rows × rows and run SGEMM
    let mut c = vec![0.0f32; rows * rows];

    unsafe {
        // C ← 1 · A · Aᵀ + 0 · C
        //
        // row‑major layout ⇒ row_stride = cols, col_stride = 1
        // Aᵀ therefore has row_stride = 1, col_stride = cols
        sgemm(
            rows, // m
            cols, // k
            rows, // n
            1.0,  // α
            a.as_ptr(),
            cols as isize,
            1, // A
            a.as_ptr(),
            1,
            cols as isize, // B = Aᵀ
            0.0,           // β
            c.as_mut_ptr(),
            rows as isize,
            1, // C (row‑major)
        );
    }
    // NOTE: enable the "threading" feature on matrixmultiply if you want it
    // to spawn worker threads internally (reads MATMUL_NUM_THREADS env‑var):contentReference[oaicite:0]{index=0}.

    // 3.  Re‑pack C back into your SIMD/TinyVec layout
    Ok(flatten_vec_to_wide_matrix(&c, rows, rows)?)
}

/// Triangular cosine‑similarity, SIMD + Rayon.
///
/// * Rows are L2‑normalised first (still SIMD).
/// * For each `i` we compute `sim(i, j)` only for `j ≥ i`.
///   After all threads finish we mirror the values to the lower triangle.
/// * Returns a dense `WideMatrix` laid out like the input helpers
///   (`flatten_vec_to_wide_matrix` packs the flat `Vec<f32>` into SIMD).
pub fn similarity_matrix_wide_opt_par(p0: &WideMatrix) -> Result<WideMatrix> {
    let n = p0.len();
    if n == 0 {
        return Ok(SmallVec::new());
    }
    let mut normed = p0.clone();
    normalize_l2_wide_new(&mut normed);
    

    // 1. pre‑allocate the whole N×N buffer
    let mut sims = vec![0.0f32; n * n];

    // 2. fill one row per Rayon job (upper triangle + diagonal)
    sims.par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row_slice)| {
            row_slice[i] = 1.0; // diagonal
            let row_vec = &normed[i];

            for j in (i + 1)..n {
                let sim = dot_wide(row_vec, &normed[j]);
                row_slice[j] = sim; // write only j ≥ i
            }
        });

    // 3. single‑threaded mirror to the lower triangle
    for i in 0..n {
        for j in (i + 1)..n {
            sims[j * n + i] = sims[i * n + j];
        }
    }

    // 4. pack the flat buffer back into a WideMatrix
    Ok(flatten_vec_to_wide_matrix(&sims, n, n)?)
}

pub fn create_markov_matrix_discrete_wide(
    weights: &WideMatrix,
    threshold: f32,
) -> Result<WideMatrix> {
    if weights.is_empty() {
        return Ok(SmallVec::new());
    }

    let discrete_weights = wide_discrete_weights(weights, threshold)?;
    wide_create_markov_matrix(&discrete_weights)
}

fn wide_discrete_weights(weights: &WideMatrix, threshold: f32) -> Result<WideMatrix> {
    let th = Wide::splat(threshold);
    let ones = Wide::splat(1.0); // Wide with all lanes set to 1.0
    let zeros = Wide::splat(0.0);
    let mut discrete_weights = SmallVec::with_capacity(weights.len());
    for row in weights {
        let mut bin_row: WideRow = SmallVec::with_capacity(row.len());
        for &w in row {
            // cmp_ge returns a “mask” vector; blend chooses per‑lane
            let bin = w.simd_ge(th).select(ones, zeros); // if w >= th then 1.0 else 0.0
            bin_row.push(bin);
        }
        discrete_weights.push(bin_row);
    }
    Ok(discrete_weights)
}

pub fn wide_create_markov_matrix(weights_matrix: &WideMatrix) -> Result<WideMatrix> {
    let min = min_wide_matrix(weights_matrix).unwrap();
    if min <= 0.0 {
        Ok(wide_softmax(weights_matrix)?)
    } else {
        let row_sum = wide_matrix_row_sum(weights_matrix);
        let mut out = SmallVec::with_capacity(weights_matrix.len());
        for row in weights_matrix {
            let sum = wide_row_sum(row)?;
            let inv_sum = Wide::splat(1.0 / sum);
            let mut exp_row: WideRow = SmallVec::with_capacity(row.len());
            for &w in row {
                let nw = w * inv_sum;
                exp_row.push(nw);
            }
            out.push(exp_row);
        }
        //println!("create_markov_matrix row_sum {:?}", row_sum.shape());
        Ok(out)
    }
}

pub fn wide_row_sum(row: &WideRow) -> Result<f32> {
    let mut sum = 0.0f32;
    for w in row {
        sum += w.reduce_sum();
    }
    Ok(sum)
}

pub fn wide_matrix_row_sum(mat: &WideMatrix) -> Result<SmallVec<[f32; LANES]>> {
    let mut out = SmallVec::with_capacity(mat.len());

    for row in mat {
        let mut sum = 0.0f32;
        for w in row {
            sum += w.reduce_sum();
        }
        out.push(sum);
    }
    Ok(out)
}

#[inline(always)]
fn dot_wide(a: &WideRow, b: &WideRow) -> f32 {
    let mut acc = Wide::splat(0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc.add_assign( x * y) // element‑wise multiply and lane‑wise sum
    }
    acc.reduce_sum() // lane‑wise sum → scalar
}

#[inline(always)]
fn dot_wide_par(a: &WideRow, b: &WideRow) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // heuristic: ≤ 32 registers ≈ 128 floats – cheaper to stay serial
    if a.len() <= 32 {
        let mut acc = Wide::splat(0.0);
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc += x * y;
        }
        return acc.reduce_sum();
    }

    // parallel path ---------------------------------------------------------
    let acc = a
        .par_iter() // &Wide → ParallelIterator
        .zip(b.par_iter()) // pair up the registers
        .map(|(&x, &y)| x * y) // element‑wise product
        .reduce(|| Wide::splat(0.0), |s, v| s + v);

    acc.reduce_sum() // single horizontal reduction → scalar
}

#[inline(always)]
fn l2_norm(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum.sqrt()
}

/// C = A·B  (all three are *transposed* matrices: rows = columns)
fn square_transposed(mat: &WideMatrix) -> WideMatrix {
    let n = mat.len();
    let mut out = SmallVec::with_capacity(n);

    for i in 0..n {
        let mut row_vals = Vec::with_capacity(n);
        for j in 0..n {
            row_vals.push(dot_wide(&mat[i], &mat[j]));
        }
        out.push(vec_to_row(&row_vals));
    }
    out
}

/// y = A·x  (A is *transposed*)
#[inline(always)]
fn mat_vec_mul(w_mat: &WideMatrix, w_row: &WideRow) -> WideRow {
    let res_vec: Vec<_> = w_mat.iter().map(|m_row| dot_wide(m_row, w_row)).collect();
    vec_to_row(&res_vec)
}

#[inline(always)]
fn row_sub(a: &WideRow, b: &WideRow) -> WideRow {
    a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
}

// ────────────────────────────────────────────────────────────
//            Power iteration, wide version
// ────────────────────────────────────────────────────────────

/// Power‑method eigenvector for a (row‑stochastic) transition matrix.
///
/// * `transition_matrix` must be **transposed**: each row is a column of the
///   original matrix.  This matches the layout returned by the earlier helper
///   that converts an `Array2` with `.t()`.
/// * Returns the dominant left eigenvector as a SIMD‑packed `WideRow`.
pub fn power_method_wide(
    transition_matrix: &WideMatrix,
    increase_power: bool,
    max_iter: usize,
) -> Result<WideRow> {
    let n = transition_matrix.len();
    if n == 0 {
        return Err(anyhow!("matrix has zero size"));
    }
    if n == 1 {
        return Ok(vec_to_row(&[1.0]));
    }

    // initial vector: all‑ones, normalised implicitly because rows are stochastic
    let mut eig_wide = wide_row_ones(n);
    let mut trans = transition_matrix.clone(); // still transposed

    for _ in 0..max_iter {
        let eig_next = mat_vec_mul(&trans, &eig_wide);
        let sub_row = row_sub(&eig_next, &eig_wide);
        let norm = norm_wide(&sub_row);
        // convergence check (L2 distance)
        if norm < 1e-5 {
            return Ok(eig_next);
        }

        eig_wide = eig_next;

        // optional acceleration: square the matrix each round
        if increase_power {
            trans = square_transposed(&trans);
        }
    }

    Ok(eig_wide)
}

#[inline(always)]
fn wide_size(n: usize) -> usize {
    // round up to the next multiple of LANES
    (n + LANES - 1) / LANES
}

#[inline(always)]
fn wide_row_ones(n: usize) -> WideRow {
    let n_wides = wide_size(n); // round up to the next multiple of LANES
    (0..n_wides)
        .map(|_| Wide::splat(1.0)) // create a Wide with all lanes set to 1.0
        .collect()
}

pub fn wide_softmax(mat: &WideMatrix) -> Result<WideMatrix> {
    let mut out = SmallVec::with_capacity(mat.len());

    for row in mat {
        // 1. exponentiate every element and accumulate the row sum
        let mut exp_row: WideRow = SmallVec::with_capacity(row.len());
        let mut sum = 0.0f32;

        for &w in row {
            // wide::f32x4 has `exp` (element‑wise) and `horizontal_sum`
            let ew = w * w; // element‑wise square
            sum += ew.reduce_sum();
            exp_row.push(ew);
        }

        if !sum.is_finite() || sum == 0.0 {
            return Err(anyhow!("row has zero or non‑finite exp sum"));
        }

        // 2. normalise: multiply each SIMD register by 1/sum
        let inv_sum = Wide::splat(1.0 / sum);
        for w in &mut exp_row {
            *w = *w * inv_sum; // element‑wise multiply
        }

        out.push(exp_row);
    }

    Ok(out)
}

pub fn lexrank_wide(
    embeddings: &Vec<Vec<f32>>,
    threshold: Option<f32>,
    max_iter: usize,
) -> Result<Vec<(usize, f32)>> {
    if embeddings.is_empty() {
        return Ok(vec![]);
    }
    let embeddings_flatten: Vec<f32> = embeddings.iter().flatten().cloned().collect();
    let n_rows = embeddings.len();
    let n_cols = embeddings[0].len();
    let embeddings_matrix = vec_to_wide_matrix(embeddings)?;
    lexrank_ts_wide(&embeddings_matrix, threshold, max_iter)
}

#[inline(always)]
fn row_to_vec(row: &WideRow) -> SmallVec<[f32; LANES]> {
    row.iter().flat_map(|w| w.to_array()).collect()
}

pub fn lexrank_ts_wide(
    embeddings: &WideMatrix,
    threshold: Option<f32>,
    max_iter: usize,
) -> Result<Vec<(usize, f32)>> {
    if embeddings.is_empty() {
        return Ok(vec![]);
    }
    let sim_matrix = similarity_matrix_wide_opt(&embeddings)?;
    let threshold = threshold.map(|threshold| {
        let sim_min: f32 = min_wide_matrix(embeddings).unwrap();
        //println!("sim_min: {:8.16}", sim_min);
        let sim_range = 1f32 - sim_min;
        sim_min + threshold * sim_range
    });
    let scores = degree_centrality_scores_wide(&sim_matrix, false, threshold, max_iter, true)?;
    let scores_vec = row_to_vec(&scores);
    let mut ranked_sentences: Vec<_> = (0..embeddings.len()).zip(scores_vec).collect();
    ranked_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(ranked_sentences)
}

fn row_div(row: &WideRow, denom: f32) -> WideRow {
    let scale_wide = Wide::splat(denom);
    row.iter().map(|&w| w / scale_wide).collect()
}

pub fn stationary_distribution_wide(
    transition_matrix: &WideMatrix,
    increase_power: bool,
    max_iter: usize,
    normalized: bool,
) -> Result<WideRow> {
    let n_rows = transition_matrix.len();
    let mut distribution = power_method_wide(&transition_matrix, increase_power, max_iter)?;

    if normalized {
        distribution = row_div(&distribution, n_rows as f32);
    }

    Ok(distribution)
}

pub fn degree_centrality_scores_wide(
    similarity_matrix: &WideMatrix,
    increase_power: bool,
    threshold: Option<f32>,
    max_iter: usize,
    normalized: bool,
) -> Result<WideRow> {
    if threshold.is_some() {
        let threshold = threshold.unwrap();
        assert!(
            threshold >= 0.0 && threshold < 1.0,
            "'threshold' should be a floating-point number from the interval [0, 1) or None"
        );
    }
    let markov_matrix = if let Some(threshold) = threshold {
        create_markov_matrix_discrete_wide(&similarity_matrix, threshold)?
    } else {
        wide_create_markov_matrix(similarity_matrix)?
    };

    let scores =
        stationary_distribution_wide(&markov_matrix, increase_power, max_iter, normalized)?;

    Ok(scores)
}
