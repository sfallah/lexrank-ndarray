use criterion::{black_box, criterion_main, Criterion};
use ndarray_benches::{get_rand_arr1_f32, get_rand_arr2_f32, linear_forward};

pub fn ndarray_linear_forward(c: &mut Criterion) {
    let m = 16;
    let n = 384;
    let k = 1536;
    let mean = 100.0;
    let std_dev = 15.0;
    let lhs = get_rand_arr2_f32(m, k, mean, std_dev).unwrap();
    let rhs = get_rand_arr2_f32(k, n, mean, std_dev).unwrap();
    let bias = get_rand_arr1_f32(n, mean, std_dev).unwrap();
    c.bench_function("ndarray_linear_forward", |b| {
        b.iter(|| {
            let result =
                linear_forward(black_box(&lhs), black_box(&rhs), black_box(&bias)).unwrap();
            black_box(result);
        });
    });
}

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(20))
        .configure_from_args();
    ndarray_linear_forward(&mut criterion);
}

criterion_main!(benches);
