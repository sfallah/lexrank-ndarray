use criterion::{criterion_main, Criterion};
use lexrank_ndarray::testing::{
    array2_from_vec, load_split_tensor, load_split_vec, load_splits_data,
};
use lexrank_ndarray::{flatten_vec_to_wide_matrix, lexrank_array, normalize_l2, normalize_l2_par, normalize_l2_wide, normalize_l2_wide_par, similarity_matrix, similarity_matrix_mm, similarity_matrix_par, similarity_matrix_par_new, similarity_matrix_wide, similarity_matrix_wide_opt, similarity_matrix_wide_opt_par, similarity_matrix_wide_par};
#[cfg(feature = "accelerate")]
use lexrank_ndarray::cosine_f32_matrix;
use rayon::prelude::*;
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};
use rayon::ThreadPoolBuilder;

pub fn ndarray_normalize_l2(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("ndarray_normalize_l2 {}", dataset).as_str(), |b| {
        b.iter(|| {
            black_box(embeds_vec.par_iter()).for_each(|(shape, vec)| {
                let embeddings = array2_from_vec(vec, shape).unwrap();
                let result = normalize_l2(black_box(&embeddings)).unwrap();
                black_box(result);
            });
        });
    });
}

pub fn ndarray_normalize_l2_par(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(
        format!("ndarray_normalize_l2_par {}", dataset).as_str(),
        |b| {
            b.iter(|| {
                black_box(embeds_vec.par_iter()).for_each(|(shape, vec)| {
                    let mut embeddings = array2_from_vec(vec, shape).unwrap();
                    normalize_l2_par(black_box(&mut embeddings));
                });
            });
        },
    );
}
pub fn wide_normalize_l2_par(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("wide_normalize_l2_par {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let embeddings = flatten_vec_to_wide_matrix(vec, shape[0], shape[1]).unwrap();
                let result = normalize_l2_wide_par(black_box(&embeddings)).unwrap();
                black_box(result);
            });
        });
    });
}

pub fn wide_normalize_l2(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("wide_normalize_l2 {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let embeddings = flatten_vec_to_wide_matrix(vec, shape[0], shape[1]).unwrap();
                let result = normalize_l2_wide(black_box(&embeddings)).unwrap();
                black_box(result);
            });
        });
    });
}

pub fn ndarray_cosine_sim(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("ndarray_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let embeddings = array2_from_vec(vec, shape).unwrap();
                let result = similarity_matrix(black_box(&embeddings)).unwrap();
                black_box(result);
            });
        });
    });
}

pub fn ndarray_cosine_sim_par(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("ndarray_cosine_sim_par {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let mut embeddings = array2_from_vec(vec, shape).unwrap();
                similarity_matrix_par(black_box(&mut embeddings));
            });
        });
    });
}

pub fn ndarray_cosine_sim_par_new(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("ndarray_cosine_sim_par_new {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let embeddings = array2_from_vec(vec, shape).unwrap();
                let sims = similarity_matrix_par_new(black_box(&embeddings)).unwrap();
                black_box(sims);
            });
        });
    });
}

pub fn wide_cosine_sim(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("wide_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let embeddings = flatten_vec_to_wide_matrix(vec, shape[0], shape[1]).unwrap();
                let result = similarity_matrix_wide_opt_par(black_box(&embeddings)).unwrap();
                black_box(result);
            });
        });
    });
}

#[cfg(feature = "accelerate")]
pub fn accelerate_cosine_sim(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
    c.bench_function(format!("accelerate_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|(shape, vec)| {
                let result = cosine_f32_matrix(black_box(vec), shape[0], shape[1]);
                black_box(result);
            });
        });
    });
}

pub fn ndarray_lexrank(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_tensor(tensor_file, split).unwrap())
        .collect();
    let embeds_vec: Vec<_> = embeds_vec
        .iter()
        .map(|embed| {
            let embed_flatten: Vec<f32> = embed.flatten().to_vec();
            let shape = embed.shape();
            (embed_flatten, shape[0], shape[1])
        })
        .collect();
    println!("embeds_vec {:?}", embeds_vec.len());
    c.bench_function(format!("ndarray_lexrank {}", dataset).as_str(), |b| {
        b.iter_custom(|iters| {
            let value = embeds_vec.clone();
            thread::spawn(move || {
                let mut duration = Duration::new(0, 0);
                for _ in 0..iters {
                    let start = Instant::now();
                    value.par_iter().for_each(|(embed, no_seq, embd_dim)| {
                        let scores = lexrank_array(embed, *no_seq, *embd_dim, None, 10000).unwrap();
                        black_box(scores);
                    });
                    let elapsed = start.elapsed();
                    duration = duration.checked_add(elapsed).unwrap();
                }
                duration
            })
            .join()
            .unwrap()
        });
    });
}

pub fn benches() {
    let num_threads = num_cpus::get_physical(); // Replace with the number of threads you want

    // Set the custom ThreadPool as the global Rayon runtime
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();


    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
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
        // Ndarray Normalization
        //ndarray_normalize_l2(&mut criterion, dataset, tensor_file);
        //ndarray_normalize_l2_par(&mut criterion, dataset, tensor_file);

        // Wide Normalization
        //wide_normalize_l2(&mut criterion, dataset, tensor_file);
        //wide_normalize_l2_par(&mut criterion, dataset, tensor_file);

        // Cosine Similarity
        ndarray_cosine_sim(&mut criterion, dataset, tensor_file);
        ndarray_cosine_sim_par(&mut criterion, dataset, tensor_file);
        ndarray_cosine_sim_par_new(&mut criterion, dataset, tensor_file);

        #[cfg(feature = "accelerate")]
        accelerate_cosine_sim(&mut criterion, dataset, tensor_file);

        //wide_cosine_sim(&mut criterion, dataset, tensor_file);
        //ndarray_lexrank(&mut criterion, dataset, tensor_file);
    }
}

criterion_main!(benches);
