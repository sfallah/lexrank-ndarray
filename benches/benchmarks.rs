use std::fmt::format;
use criterion::{black_box, criterion_main, Criterion};
use ndarray_benches::{similarity_matrix, get_rand_arr1_f32, get_rand_arr2_f32, linear_forward, lexrank_ts, normalize_l2};
use ndarray_benches::utils::{load_split_tensor, load_splits_data};
use rayon::prelude::*;

pub fn ndarray_linear_forward(c: &mut Criterion) {
    let m = 16;
    let n = 384;
    let k = 1536;
    let mean = 100.0;
    let std_dev = 15.0;
    let lhs = get_rand_arr2_f32(m, k, mean, std_dev).unwrap();
    let rhs = get_rand_arr2_f32(n, k, mean, std_dev).unwrap();
    let bias = get_rand_arr1_f32(n, mean, std_dev).unwrap();
    c.bench_function("ndarray_linear_forward", |b| {
        b.iter(|| {
            let result =
                linear_forward(black_box(&lhs), black_box(&rhs), black_box(&bias)).unwrap();
            black_box(result);
        });
    });
}

pub fn ndarray_normalize_l2(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits.iter().map(|split| load_split_tensor(tensor_file, split).unwrap()).collect();
    c.bench_function(format!("ndarray_normalize_l2 {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|embed| {
                let result = normalize_l2(black_box(embed)).unwrap();
                black_box(result);
            });
        });
    });
}
pub fn ndarray_cosine_sim(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits.iter().map(|split| load_split_tensor(tensor_file, split).unwrap()).collect();
    c.bench_function(format!("ndarray_cosine_sim {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|embed| {
                let result = similarity_matrix(black_box(embed)).unwrap();
                black_box(result);
            });
        });
    });
}
pub fn ndarray_lexrank(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits.iter().map(|split| load_split_tensor(tensor_file, split).unwrap()).collect();
    println!("embeds_vec {:?}", embeds_vec.len());
    for embed in embeds_vec.iter() {
        println!("embed {:?}", embed.shape());
    }
    c.bench_function(format!("ndarray_lexrank {}", dataset).as_str(), |b| {
        b.iter(|| {
            embeds_vec.par_iter().for_each(|embed| {
                let scores = lexrank_ts(embed, Some(0.25), 10000).unwrap();
                black_box(scores);
            });
        });
    });
}

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();

    let data_set_map = [
        ("MiniLM-L6-v2", "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/"),
        ("bge-m3", "tests/test_data/superlinear_embeddings/bge-m3/"),
        //("multilingual-e5-large-instruct", "tests/test_data/superlinear_embeddings/multilingual-e5-large-instruct/"),
    ];
    for (dataset, tensor_file) in data_set_map.iter() {
        ndarray_normalize_l2(&mut criterion, dataset, tensor_file);
        ndarray_cosine_sim(&mut criterion, dataset, tensor_file);
        ndarray_lexrank(&mut criterion, dataset, tensor_file);
    }
}

criterion_main!(benches);
