//! LexRank backend benchmarks.
//!
//! Three groups, meant to be run with different thread settings (see the environment notes on
//! each group). Every benchmark id carries the BLAS backend the binary was built with, so results
//! from several builds can be merged without losing track of which is which.
//!
//! - `kernel/<backend>/<dataset>/...`: the individual steps and the full pipeline on one thread,
//!   one split per call, cycling through every split of one fixture dataset. Times are per split.
//! - `scaling/<backend>/d768/...`: the same functions on synthetic sentence counts, to see where
//!   the backends and BLAS threading start to matter.
//! - `throughput/<backend>/threads4/...`: all fixture splits in parallel on a fixed 4-thread rayon
//!   pool, the way embedding-processing calls LexRank.
//!
//! In `kernel` and `scaling` every variant gets its input copied right before its timed call, so
//! it starts in cache, as when the pipeline copies its input or downstream hands over a freshly
//! produced embedding block. Input and output are both dropped after timing. The one copy that is
//! timed on purpose is the clone `lexrank_array` makes internally; `input-clone` measures it on
//! its own. Everything runs on a rayon worker thread, as it does downstream (see `main`).

use cblas::{sgemm, ssyrk, Layout, Part, Transpose};
use criterion::{criterion_group, BatchSize, Bencher, BenchmarkId, Criterion, SamplingMode};
use lexrank_ndarray::cblas_impl::{
    blas_cosine_f32_matrix, blas_cosine_f32_matrix_opt, blas_cosine_f32_matrix_syrk,
    blas_degree_centrality_scores, blas_lexrank_array,
};
use lexrank_ndarray::testing::{array2_from_vec, load_split_vec, load_splits_data};
use lexrank_ndarray::{degree_centrality_scores, lexrank_array, normalize_l2, similarity_matrix};
use ndarray::Array2;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

const BACKEND: &str = if cfg!(feature = "accelerate") {
    "accelerate"
} else if cfg!(feature = "blas-static") {
    "openblas-static"
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

/// A benchmark that times `routine` on one split per iteration, cycling through `splits`.
///
/// `setup` builds the iteration's input right before it is timed, so every variant starts from
/// data that is in cache. `routine` returns what it was given along with its result, so neither
/// the input nor the output is freed inside the timed region.
fn per_split<'a, I, O>(
    splits: &'a [Split],
    mut setup: impl FnMut(usize) -> I + 'a,
    mut routine: impl FnMut(I, &Split) -> O + 'a,
) -> impl FnMut(&mut Bencher<'_>) + 'a {
    move |b| {
        let mut next = 0;
        b.iter_batched(
            || {
                let k = next % splits.len();
                next += 1;
                (setup(k), k)
            },
            |(input, k)| routine(black_box(input), &splits[k]),
            BatchSize::PerIteration,
        )
    }
}

/// Run with one thread everywhere: `RAYON_NUM_THREADS=1` and the BLAS thread variable set to 1
/// (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`).
/// The parts add up to the whole: `lexrank-ndarray_api` is `input-clone` + `similarity-ndarray`
/// + `centrality-ndarray`, and `lexrank-cblas` is `similarity-cblas_sdot` + `centrality-cblas`,
/// each plus a final sort.
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
        let flat = |k: usize| splits[k].flat.clone();

        let mut group = c.benchmark_group(format!("kernel/{BACKEND}/{dataset}"));
        group.bench_function(
            "similarity-ndarray",
            per_split(
                &splits,
                |k| splits[k].array.clone(),
                |mut a, _| {
                    let sim = similarity_matrix(&mut a);
                    (a, sim)
                },
            ),
        );
        group.bench_function(
            "similarity-cblas_sdot",
            per_split(&splits, flat, |v, s| {
                let sim = blas_cosine_f32_matrix(&v, s.rows, s.cols);
                (v, sim)
            }),
        );
        group.bench_function(
            "similarity-cblas_sgemm",
            per_split(&splits, flat, |v, s| {
                let sim = blas_cosine_f32_matrix_opt(&v, s.rows, s.cols);
                (v, sim)
            }),
        );
        group.bench_function(
            "centrality-ndarray",
            per_split(
                &splits,
                |k| nd_sims[k].clone(),
                |sim, _| {
                    let scores = degree_centrality_scores(&sim, false, None, MAX_ITER, true);
                    (sim, scores.unwrap())
                },
            ),
        );
        group.bench_function(
            "centrality-cblas",
            per_split(
                &splits,
                |k| blas_sims[k].clone(),
                |sim, s| {
                    let scores =
                        blas_degree_centrality_scores(&sim, s.rows, false, None, MAX_ITER, true);
                    (sim, scores.unwrap())
                },
            ),
        );
        group.bench_function(
            "lexrank-ndarray_api",
            per_split(&splits, flat, |v, s| {
                let ranked = lexrank_array(&v, s.rows, s.cols, None, MAX_ITER);
                (v, ranked.unwrap())
            }),
        );
        group.bench_function(
            "lexrank-cblas",
            per_split(&splits, flat, |v, s| {
                let ranked = blas_lexrank_array(&v, s.rows, s.cols, None, MAX_ITER);
                (v, ranked.unwrap())
            }),
        );
        group.bench_function(
            "input-clone",
            per_split(&splits, flat, |v, _| {
                let copy = v.clone();
                (v, copy)
            }),
        );
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
        let splits = [synthetic_split(rows, DIM, rows as u64)];
        let flat = |k: usize| splits[k].flat.clone();

        group.bench_function(
            BenchmarkId::new("similarity-ndarray", rows),
            per_split(
                &splits,
                |k| splits[k].array.clone(),
                |mut a, _| {
                    let sim = similarity_matrix(&mut a);
                    (a, sim)
                },
            ),
        );
        group.bench_function(
            BenchmarkId::new("similarity-cblas_sdot", rows),
            per_split(&splits, flat, |v, s| {
                let sim = blas_cosine_f32_matrix(&v, s.rows, s.cols);
                (v, sim)
            }),
        );
        group.bench_function(
            BenchmarkId::new("similarity-cblas_sgemm", rows),
            per_split(&splits, flat, |v, s| {
                let sim = blas_cosine_f32_matrix_opt(&v, s.rows, s.cols);
                (v, sim)
            }),
        );
        group.bench_function(
            BenchmarkId::new("lexrank-ndarray_api", rows),
            per_split(&splits, flat, |v, s| {
                let ranked = lexrank_array(&v, s.rows, s.cols, None, MAX_ITER);
                (v, ranked.unwrap())
            }),
        );
        group.bench_function(
            BenchmarkId::new("lexrank-cblas", rows),
            per_split(&splits, flat, |v, s| {
                let ranked = blas_lexrank_array(&v, s.rows, s.cols, None, MAX_ITER);
                (v, ranked.unwrap())
            }),
        );
    }
    group.finish();
}

/// The full Gram matrix `E·Eᵀ` through one `sgemm`, without normalisation.
fn gram_sgemm(e: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (n, k) = (rows as i32, cols as i32);
    let mut gram = vec![0.0f32; rows * rows];
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            n,
            n,
            k,
            1.0,
            e,
            k,
            e,
            k,
            0.0,
            &mut gram,
            n,
        )
    };
    gram
}

/// The upper triangle of the Gram matrix `E·Eᵀ` through one `ssyrk`, without normalisation.
fn gram_ssyrk(e: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (n, k) = (rows as i32, cols as i32);
    let mut gram = vec![0.0f32; rows * rows];
    unsafe {
        ssyrk(
            Layout::RowMajor,
            Part::Upper,
            Transpose::None,
            n,
            k,
            1.0,
            e,
            k,
            0.0,
            &mut gram,
            n,
        )
    };
    gram
}

/// Split the similarity step into its parts and compare it with one-call alternatives.
/// Run single-threaded like `kernel_benches` (and optionally again with the BLAS default
/// threads). Before timing, `blas_cosine_f32_matrix_syrk` is checked against `similarity_matrix`.
///
/// - `normalize-ndarray` + `gram-ndarray_dot` make up `total-ndarray` (`similarity_matrix`).
/// - `gram-sgemm` is the same full Gram matrix straight through CBLAS, without ndarray.
/// - `gram-ssyrk` computes only its upper triangle; `total-ssyrk` adds the norm scaling.
/// - `norms-snrm2` is `snrm2` per row, what `blas_cosine_f32_matrix` used before it took
///   norms from `sdot`; `total-cblas_sdot` is that function as it is now.
fn similarity_benches(c: &mut Criterion) {
    let mut inputs: Vec<(String, Vec<Split>)> = [DATASETS[0], DATASETS[3]]
        .iter()
        .map(|(name, path)| (name.to_string(), load_dataset(path)))
        .collect();
    for cols in [384, 768, 1536] {
        for rows in [16, 24, 32, 48, 64, 128] {
            let split = synthetic_split(rows, cols, (rows * cols) as u64);
            inputs.push((format!("d{cols}-n{rows}"), vec![split]));
        }
    }

    for (name, splits) in &inputs {
        for s in splits {
            let expected = similarity_matrix(&mut s.array.clone());
            let got = blas_cosine_f32_matrix_syrk(&s.flat, s.rows, s.cols);
            let diff = expected
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(diff <= 1e-5, "similarity_ssyrk differs by {diff} on {name}");
        }
        let normalized: Vec<Array2<f32>> = splits
            .iter()
            .map(|s| {
                let mut a = s.array.clone();
                normalize_l2(&mut a);
                a
            })
            .collect();
        let flat = |k: usize| splits[k].flat.clone();

        let mut group = c.benchmark_group(format!("similarity/{BACKEND}/{name}"));
        if splits.len() == 1 {
            group
                .sampling_mode(SamplingMode::Flat)
                .sample_size(10)
                .measurement_time(Duration::from_secs(2));
        }
        group.bench_function(
            "normalize-ndarray",
            per_split(
                splits,
                |k| splits[k].array.clone(),
                |mut a, _| {
                    normalize_l2(&mut a);
                    a
                },
            ),
        );
        group.bench_function(
            "gram-ndarray_dot",
            per_split(
                splits,
                |k| normalized[k].clone(),
                |a, _| {
                    let gram = a.dot(&a.t());
                    (a, gram)
                },
            ),
        );
        group.bench_function(
            "total-ndarray",
            per_split(
                splits,
                |k| splits[k].array.clone(),
                |mut a, _| {
                    let sim = similarity_matrix(&mut a);
                    (a, sim)
                },
            ),
        );
        group.bench_function(
            "gram-sgemm",
            per_split(splits, flat, |v, s| {
                let gram = gram_sgemm(&v, s.rows, s.cols);
                (v, gram)
            }),
        );
        group.bench_function(
            "gram-ssyrk",
            per_split(splits, flat, |v, s| {
                let gram = gram_ssyrk(&v, s.rows, s.cols);
                (v, gram)
            }),
        );
        group.bench_function(
            "total-ssyrk",
            per_split(splits, flat, |v, s| {
                let sim = blas_cosine_f32_matrix_syrk(&v, s.rows, s.cols);
                (v, sim)
            }),
        );
        group.bench_function(
            "norms-snrm2",
            per_split(splits, flat, |v, s| {
                let norms: Vec<f32> = v
                    .chunks_exact(s.cols)
                    .map(|row| unsafe { cblas::snrm2(s.cols as i32, row, 1) })
                    .collect();
                (v, norms)
            }),
        );
        group.bench_function(
            "total-cblas_sdot",
            per_split(splits, flat, |v, s| {
                let sim = blas_cosine_f32_matrix(&v, s.rows, s.cols);
                (v, sim)
            }),
        );
        group.finish();
    }
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
    targets = kernel_benches, scaling_benches, throughput_benches, similarity_benches
}

/// `criterion_main!`, but run on a rayon worker thread. `blas_cosine_f32_matrix` uses rayon
/// internally; called from a thread outside the pool, every call hands its work to a pool thread
/// and sleeps until it is done, even with one rayon thread, which adds wake-up latency and jitter
/// to exactly the CBLAS variants. embedding-processing calls LexRank from inside rayon workers,
/// where that handoff does not happen. The pool honours `RAYON_NUM_THREADS`.
fn main() {
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(16 << 20)
        .build()
        .unwrap();
    pool.install(|| {
        benches();
        Criterion::default().configure_from_args().final_summary();
    });
}
