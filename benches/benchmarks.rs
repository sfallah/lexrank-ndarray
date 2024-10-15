use criterion::{black_box, criterion_main, Criterion};
use lexrank_ndarray::testing::{
    array2_from_vec, load_split_tensor, load_split_vec, load_splits_data,
};
use lexrank_ndarray::{lexrank_ts, normalize_l2, similarity_matrix};
use rayon::prelude::*;

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
pub fn ndarray_lexrank(c: &mut Criterion, dataset: &str, tensor_file: &str) {
    let splits = load_splits_data(tensor_file).unwrap();
    let embeds_vec: Vec<_> = splits
        .iter()
        .map(|split| load_split_tensor(tensor_file, split).unwrap())
        .collect();
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
        (
            "MiniLM-L6-v2",
            "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/",
        ),
        //("bge-m3", "tests/test_data/superlinear_embeddings/bge-m3/"),
        //("multilingual-e5-large-instruct", "tests/test_data/superlinear_embeddings/multilingual-e5-large-instruct/"),
    ];
    for (dataset, tensor_file) in data_set_map.iter() {
        //ndarray_normalize_l2(&mut criterion, dataset, tensor_file);
        //ndarray_cosine_sim(&mut criterion, dataset, tensor_file);
        ndarray_lexrank(&mut criterion, dataset, tensor_file);
    }
}

criterion_main!(benches);
