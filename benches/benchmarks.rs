use criterion::{black_box, criterion_main, Criterion};
use ndarray_benches::{similarity_matrix, get_rand_arr1_f32, get_rand_arr2_f32, linear_forward, lexrank_ts};
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

pub fn ndarray_cosine_sim(c: &mut Criterion) {
    let m = 30;
    let n = 384;
    let mean = 100.0;
    let std_dev = 15.0;
    let embeddings = get_rand_arr2_f32(m, n, mean, std_dev).unwrap();
    c.bench_function("ndarray_cosine_sim", |b| {
        b.iter(|| {
            let result =
                similarity_matrix(black_box(&embeddings)).unwrap();
            black_box(result);
        });
    });
}

pub fn ndarray_lexrank(c: &mut Criterion) {
    let tensor_file = "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/";
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits.iter().map(|split| load_split_tensor(tensor_file, split).unwrap()).collect();
    c.bench_function("ndarray_lexrank", |b| {
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
    //ndarray_cosine_sim(&mut criterion);
    //ndarray_linear_forward(&mut criterion);
    ndarray_lexrank(&mut criterion);
}

criterion_main!(benches);
