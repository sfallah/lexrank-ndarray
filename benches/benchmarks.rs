//! LexRank backend benchmarks.
//!
//! Three groups, meant to be run with different thread settings (see the environment notes on
//! each group). Every benchmark id carries the BLAS backend the binary was built with, so results
//! from several builds can be merged without losing track of which is which.
//!
//! - `kernel/<backend>/<dataset>/...`: the cost of the individual steps and of the full pipeline,
//!   one call at a time on one thread, summed over every split of one fixture dataset.
//! - `scaling/<backend>/d768/...`: the same functions on synthetic sentence counts, to see where
//!   the backends and BLAS threading start to matter.
//! - `throughput/<backend>/threads4/...`: all fixture splits in parallel on a fixed 4-thread rayon
//!   pool, the way embedding-processing calls LexRank.
//!
//! All inputs are built before timing, and outputs are dropped after it (`iter_batched`), so the
//! ndarray and CBLAS variants are timed on the same terms. The one copy that is timed on purpose
//! is the input clone `lexrank_array` makes internally; `input-clone` measures it on its own.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, SamplingMode};
use lexrank_ndarray::cblas_impl::{
    blas_cosine_f32_matrix, blas_cosine_f32_matrix_opt, blas_degree_centrality_scores,
    blas_lexrank_array,
};
use lexrank_ndarray::testing::{array2_from_vec, load_split_vec, load_splits_data};
use lexrank_ndarray::{degree_centrality_scores, lexrank_array, similarity_matrix};
use ndarray::Array2;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

const BACKEND: &str = if cfg!(feature = "accelerate") {
    "accelerate"
} else if cfg!(feature = "mkl") {
    "mkl"
} else if cfg!(feature = "blis") {
    "blis"
} else if cfg!(feature = "blas") {
    "openblas"
} else {
    "none"
};

const DATASETS: [(&str, &str); 4] = [
    (
        "MiniLM-d384",
        "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2",
    ),
    (
        "snowflake-d768",
        "tests/test_data/superlinear_embeddings/snowflake-arctic-embed-m-v1.5",
    ),
    (
        "bge-d1024",
        "tests/test_data/superlinear_embeddings/bge-reranker-v2",
    ),
    (
        "gte-Qwen2-d1536",
        "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct",
    ),
];

const MAX_ITER: usize = 10000;

struct Split {
    rows: usize,
    cols: usize,
    flat: Vec<f32>,
    array: Array2<f32>,
}

impl Split {
    fn new(rows: usize, cols: usize, flat: Vec<f32>) -> Self {
        let array = array2_from_vec(&flat, &vec![rows, cols]).unwrap();
        Split {
            rows,
            cols,
            flat,
            array,
        }
    }
}

fn load_dataset(path: &str) -> Vec<Split> {
    load_splits_data(path)
        .unwrap()
        .iter()
        .map(|split| {
            let (shape, flat) = load_split_vec(path, split).unwrap();
            Split::new(shape[0], shape[1], flat)
        })
        .collect()
}

/// Deterministic embeddings shaped like the fixtures: a shared direction plus per-sentence noise,
/// so pairwise cosine similarities are positive, around 0.4, and the Markov matrix takes the same
/// row-normalisation path as real embeddings.
fn synthetic_split(rows: usize, cols: usize, seed: u64) -> Split {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let base: Vec<f32> = (0..cols).map(|_| next()).collect();
    let flat = (0..rows * cols)
        .map(|i| 0.8 * base[i % cols] + next())
        .collect();
    Split::new(rows, cols, flat)
}

/// Run with one thread everywhere: `RAYON_NUM_THREADS=1` and the BLAS thread variable set to 1
/// (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`).
/// Each iteration processes every split of the dataset once, sequentially.
fn kernel_benches(c: &mut Criterion) {
    for (dataset, path) in DATASETS {
        let splits = load_dataset(path);
        let nd_sims: Vec<Array2<f32>> = splits
            .iter()
            .map(|s| similarity_matrix(&mut s.array.clone()))
            .collect();
        let blas_sims: Vec<Vec<f32>> = splits
            .iter()
            .map(|s| blas_cosine_f32_matrix(&s.flat, s.rows, s.cols))
            .collect();

        let mut group = c.benchmark_group(format!("kernel/{BACKEND}/{dataset}"));

        group.bench_function("similarity-ndarray", |b| {
            b.iter_batched(
                || splits.iter().map(|s| s.array.clone()).collect::<Vec<_>>(),
                |mut arrays| {
                    arrays
                        .iter_mut()
                        .map(|a| similarity_matrix(black_box(a)))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("similarity-cblas_sdot", |b| {
            b.iter_batched(
                || (),
                |_| {
                    splits
                        .iter()
                        .map(|s| blas_cosine_f32_matrix(black_box(&s.flat), s.rows, s.cols))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("similarity-cblas_sgemm", |b| {
            b.iter_batched(
                || (),
                |_| {
                    splits
                        .iter()
                        .map(|s| blas_cosine_f32_matrix_opt(black_box(&s.flat), s.rows, s.cols))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("centrality-ndarray", |b| {
            b.iter_batched(
                || (),
                |_| {
                    nd_sims
                        .iter()
                        .map(|sim| {
                            degree_centrality_scores(black_box(sim), false, None, MAX_ITER, true)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("centrality-cblas", |b| {
            b.iter_batched(
                || (),
                |_| {
                    blas_sims
                        .iter()
                        .zip(&splits)
                        .map(|(sim, s)| {
                            blas_degree_centrality_scores(
                                black_box(sim),
                                s.rows,
                                false,
                                None,
                                MAX_ITER,
                                true,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("lexrank-ndarray_api", |b| {
            b.iter_batched(
                || (),
                |_| {
                    splits
                        .iter()
                        .map(|s| {
                            lexrank_array(black_box(&s.flat), s.rows, s.cols, None, MAX_ITER)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("lexrank-cblas", |b| {
            b.iter_batched(
                || (),
                |_| {
                    splits
                        .iter()
                        .map(|s| {
                            blas_lexrank_array(black_box(&s.flat), s.rows, s.cols, None, MAX_ITER)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("input-clone", |b| {
            b.iter_batched(
                || (),
                |_| {
                    splits
                        .iter()
                        .map(|s| black_box(&s.flat).clone())
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

/// Run twice: once with the single-thread settings of `kernel_benches`, once with rayon and the
/// BLAS library allowed every available core, to see at which sentence count threading pays off.
fn scaling_benches(c: &mut Criterion) {
    const DIM: usize = 768;
    let mut group = c.benchmark_group(format!("scaling/{BACKEND}/d{DIM}"));
    group
        .sampling_mode(SamplingMode::Flat)
        .sample_size(10)
        .measurement_time(Duration::from_secs(3));

    for rows in [16, 64, 256, 512] {
        let split = synthetic_split(rows, DIM, rows as u64);

        group.bench_with_input(
            BenchmarkId::new("similarity-ndarray", rows),
            &split,
            |b, s| {
                b.iter_batched(
                    || s.array.clone(),
                    |mut a| similarity_matrix(black_box(&mut a)),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("similarity-cblas_sdot", rows),
            &split,
            |b, s| {
                b.iter_batched(
                    || (),
                    |_| blas_cosine_f32_matrix(black_box(&s.flat), s.rows, s.cols),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("similarity-cblas_sgemm", rows),
            &split,
            |b, s| {
                b.iter_batched(
                    || (),
                    |_| blas_cosine_f32_matrix_opt(black_box(&s.flat), s.rows, s.cols),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lexrank-ndarray_api", rows),
            &split,
            |b, s| {
                b.iter_batched(
                    || (),
                    |_| lexrank_array(black_box(&s.flat), s.rows, s.cols, None, MAX_ITER).unwrap(),
                    BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(BenchmarkId::new("lexrank-cblas", rows), &split, |b, s| {
            b.iter_batched(
                || (),
                |_| blas_lexrank_array(black_box(&s.flat), s.rows, s.cols, None, MAX_ITER).unwrap(),
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// Run with the BLAS thread variables set to 1; the rayon pool is fixed at 4 threads here, so
/// the machines are compared at equal parallelism. `downstream-lexrank_array` reproduces
/// embedding-processing's call, `lexrank_array(&embeddings.to_vec(), ..)`, copies included.
fn throughput_benches(c: &mut Criterion) {
    const THREADS: usize = 4;
    let splits: Vec<Split> = DATASETS
        .iter()
        .flat_map(|(_, path)| load_dataset(path))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap();

    let mut group = c.benchmark_group(format!("throughput/{BACKEND}/threads{THREADS}"));
    group.bench_function("downstream-lexrank_array", |b| {
        b.iter_batched(
            || (),
            |_| {
                pool.install(|| {
                    splits
                        .par_iter()
                        .map(|s| {
                            let embeddings = black_box(&s.flat).to_vec();
                            lexrank_array(&embeddings, s.rows, s.cols, None, MAX_ITER).unwrap()
                        })
                        .collect::<Vec<_>>()
                })
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("cblas-blas_lexrank_array", |b| {
        b.iter_batched(
            || (),
            |_| {
                pool.install(|| {
                    splits
                        .par_iter()
                        .map(|s| {
                            blas_lexrank_array(black_box(&s.flat), s.rows, s.cols, None, MAX_ITER)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                })
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30);
    targets = kernel_benches, scaling_benches, throughput_benches
}
criterion_main!(benches);
