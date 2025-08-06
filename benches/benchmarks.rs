use criterion::{criterion_main, Criterion};
use lexrank_ndarray::testing::{
    array2_from_vec, load_split_tensor, load_split_vec, load_splits_data,
};
use lexrank_ndarray::{lexrank_array, normalize_l2, similarity_matrix};
use rayon::prelude::*;
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

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
pub fn ndarray_cosine_sim(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_vec(tensor_file, split).unwrap())
        .collect();
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
    //println!("embeds_vec {:?}", embeds_vec.len());
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

        // Cosine Similarity
        ndarray_cosine_sim(&mut criterion, dataset, tensor_file);

        ndarray_lexrank(&mut criterion, dataset, tensor_file);
    }
}

criterion_main!(benches);
