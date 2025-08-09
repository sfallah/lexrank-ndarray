use criterion::{criterion_main, Criterion};
use lexrank_ndarray::cblas_impl::{
    blas_cosine_f32_matrix, blas_cosine_f32_matrix_opt, blas_lexrank_array, blas_softmax,
    blas_softmax_opt,
};
use lexrank_ndarray::testing::{
    array2_from_vec, load_split_tensor, load_split_vec, load_splits_data,
};
use lexrank_ndarray::{
    lexrank_array, normalize_l2, similarity_matrix, softmax, ss_cosine_f32_matrix,
};
use ndarray_rand::rand;
use rayon::prelude::*;
use std::hint::black_box;

fn ndarray_softmax_bench(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    let shape = vec![rows, cols];
    let mut embeddings = array2_from_vec(embeds, &shape).unwrap();
    let sim_matrix = similarity_matrix(&mut embeddings);
    c.bench_function("ndarray_softmax_bench", |b| {
        b.iter(|| {
            let result = softmax(black_box(&sim_matrix)).unwrap();
            black_box(result);
        });
    });
}
fn blas_softmax_bench(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    let sim_matrix = blas_cosine_f32_matrix(embeds, rows, cols);
    assert_eq!(sim_matrix.len(), rows * rows);
    c.bench_function("blas_softmax_bench", |b| {
        b.iter(|| {
            let result = blas_softmax(black_box(&sim_matrix), rows, rows);
            black_box(result);
        });
    });
}

fn blas_softmax_opt_bench(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    let sim_matrix = blas_cosine_f32_matrix_opt(embeds, rows, cols);
    assert_eq!(sim_matrix.len(), rows * rows);
    c.bench_function("blas_softmax_opt_bench", |b| {
        b.iter(|| {
            let result = blas_softmax_opt(black_box(&sim_matrix), rows, rows);
            black_box(result);
        });
    });
}

fn rand_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut embeds = vec![0f32; rows * cols];
    embeds.par_chunks_mut(cols).for_each(|chunk| {
        for i in 0..cols {
            chunk[i] = rand::random::<f32>();
        }
    });
    embeds
}

pub fn ndarray_rand_cosine_sim(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    let shape = vec![rows, cols];
    c.bench_function("ndarray_rand_cosine_sim", |b| {
        b.iter(|| {
            let mut embeddings = array2_from_vec(embeds, &shape).unwrap();
            let result = similarity_matrix(black_box(&mut embeddings));
            black_box(result);
        });
    });
}

pub fn simsimd_rand_cosine_sim(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    c.bench_function("simsimd_rand_cosine_sim", |b| {
        b.iter(|| {
            let result = ss_cosine_f32_matrix(embeds, rows, cols);
            black_box(result);
        });
    });
}

pub fn blas_rand_cosine_sim(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    let shape = vec![rows, cols];
    c.bench_function("blas_rand_cosine_sim", |b| {
        b.iter(|| {
            let result = blas_cosine_f32_matrix(black_box(embeds), rows, cols);
            black_box(result);
        });
    });
}

pub fn blas_rand_cosine_sim_opt(c: &mut Criterion, embeds: &[f32], rows: usize, cols: usize) {
    let shape = vec![rows, cols];
    c.bench_function("blas_rand_cosine_sim_opt", |b| {
        b.iter(|| {
            let result = blas_cosine_f32_matrix_opt(black_box(embeds), rows, cols);
            black_box(result);
        });
    });
}

pub fn ndarray_normalize_l2(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("ndarray_normalize_l2 {}", dataset).as_str(), |b| {
        b.iter(|| {
            black_box(embeds_vec.par_iter()).for_each(|(shape, vec)| {
                let mut embeddings = array2_from_vec(vec, shape).unwrap();
                let result = normalize_l2(black_box(&mut embeddings));
                black_box(result);
            });
        });
    });
}
pub fn ndarray_cosine_sim(
    c: &mut Criterion,
    dataset: &str,
    embeds_vec: &Vec<(Vec<usize>, Vec<f32>)>,
) {
    c.bench_function(format!("ndarray_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let mut embeddings = array2_from_vec(vec, shape).unwrap();
                let result = similarity_matrix(black_box(&mut embeddings));
                black_box(result);
            });
        });
    });
}

pub fn simsimd_cosine_sim(
    c: &mut Criterion,
    dataset: &str,
    embeds_vec: &Vec<(Vec<usize>, Vec<f32>)>,
) {
    c.bench_function(format!("simsimd_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let result = ss_cosine_f32_matrix(vec, shape[0], shape[1]);
                black_box(result);
            });
        });
    });
}

pub fn blas_cosine_sim(c: &mut Criterion, dataset: &str, embeds_vec: &Vec<(Vec<usize>, Vec<f32>)>) {
    c.bench_function(format!("blas_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let result = blas_cosine_f32_matrix(vec, shape[0], shape[1]);
                black_box(result);
            });
        });
    });
}

pub fn blas_cosine_sim_opt(
    c: &mut Criterion,
    dataset: &str,
    embeds_vec: &Vec<(Vec<usize>, Vec<f32>)>,
) {
    c.bench_function(format!("blas_cosine_sim_opt {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let result = blas_cosine_f32_matrix_opt(vec, shape[0], shape[1]);
                black_box(result);
            });
        });
    });
}

pub fn ndarray_lexrank(c: &mut Criterion, dataset: &str, embeds_vec: &Vec<(Vec<usize>, Vec<f32>)>) {
    c.bench_function(format!("ndarray_lexrank {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let scores = lexrank_array(vec, shape[0], shape[1], None, 10000).unwrap();
                black_box(scores);
            });
        });
    });
}

pub fn blas_ndarray_lexrank(
    c: &mut Criterion,
    dataset: &str,
    embeds_vec: &Vec<(Vec<usize>, Vec<f32>)>,
) {
    c.bench_function(format!("blas_ndarray_lexrank {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let scores = blas_lexrank_array(vec, shape[0], shape[1], None, 10000).unwrap();
                black_box(scores);
            });
        });
    });
}

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(40)
        .measurement_time(std::time::Duration::from_secs(40))
        .configure_from_args();

    let data_set_map = [
        //(
        //    "MiniLM-L6-v2",
        //    "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2/",
        //),
        (
            "gte-Qwen2-1.5B-instruct",
            "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct/",
        ), //("bge-m3", "tests/test_data/superlinear_embeddings/bge-m3/"),
           //("multilingual-e5-large-instruct", "tests/test_data/superlinear_embeddings/multilingual-e5-large-instruct/"),
    ];
    for (dataset, tensor_file) in data_set_map.iter() {
        let splits = load_splits_data(tensor_file).unwrap();
        let embeds_vec: Vec<_> = splits
            .iter()
            .map(|split| load_split_vec(tensor_file, split).unwrap())
            .collect();

        let rows = embeds_vec
            .iter()
            .max_by(|a, b| a.0[0].cmp(&b.0[0]))
            .unwrap()
            .0[0];
        // Random Cosine Similarity
        let run_rand_benches = true; // Set to true to run random embeddings benchmarks
        if run_rand_benches {
            let cols = embeds_vec[0].0[1]; // Dimension of the embeddings
            println!(
                "Generate rand_embeds for Dataset: {}, Rows: {}, Cols: {}",
                dataset, rows, cols
            );
            let rand_embeds: Vec<f32> = rand_matrix(rows, cols);
            //ndarray_rand_cosine_sim(&mut criterion, &rand_embeds, rows, cols);
            //simsimd_rand_cosine_sim(&mut criterion, &rand_embeds, rows, cols);
            ndarray_softmax_bench(&mut criterion, &rand_embeds, rows, cols);
            blas_softmax_bench(&mut criterion, &rand_embeds, rows, cols);
            blas_softmax_opt_bench(&mut criterion, &rand_embeds, rows, cols);

            blas_rand_cosine_sim(&mut criterion, &rand_embeds, rows, cols);
            blas_rand_cosine_sim_opt(&mut criterion, &rand_embeds, rows, cols);
        }

        let run_dataset_benches = false; // Set to true to run dataset benchmarks
        if run_dataset_benches {
            // Ndarray Normalization
            //ndarray_normalize_l2(&mut criterion, dataset, tensor_file);
            //ndarray_normalize_l2_par(&mut criterion, dataset, tensor_file);

            // Cosine Similarity
            //ndarray_cosine_sim(&mut criterion, dataset, &embeds_vec);
            //simsimd_cosine_sim(&mut criterion, dataset, &embeds_vec);
            blas_cosine_sim(&mut criterion, dataset, &embeds_vec);
            blas_cosine_sim_opt(&mut criterion, dataset, &embeds_vec);
        }

        let run_lexrank_benches = false; // Set to true to run lexrank benchmarks
        if run_lexrank_benches {
            // LexRank
            ndarray_lexrank(&mut criterion, dataset, &embeds_vec);
            blas_ndarray_lexrank(&mut criterion, dataset, &embeds_vec);
        }
    }
}

criterion_main!(benches);
