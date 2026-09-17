#[cfg(test)]
pub mod tests {
    use lexrank_ndarray::cblas_impl::{
        blas_cosine_f32_matrix, blas_cosine_f32_matrix_opt, blas_lexrank_array, blas_softmax,
    };
    use lexrank_ndarray::testing::{
        array2_from_vec, f32_close, get_rand_arr1_f32, get_rand_arr2_f32, load_split_tensor,
        load_split_vec, load_splits_data,
    };
    use lexrank_ndarray::{
        cos_similarity, lexrank_array, normalize_l2, similarity_matrix, softmax,
    };
    use ndarray::{array, Array1, Array2, Axis};
    use rayon::prelude::ParallelSliceMut;
    use rayon::prelude::*;
    use simsimd::SpatialSimilarity;

    #[test]
    fn ndarray_rnd_cosine_sim() -> anyhow::Result<()> {
        let m = 25;
        let n = 384;
        let mean = 100.0;
        let std_dev = 15.0;
        let mut embeddings = get_rand_arr2_f32(m, n, mean, std_dev)?;
        let sim_matrix = similarity_matrix(&mut embeddings);
        assert_eq!(sim_matrix.shape(), [m, m]);
        println!("{:8.16}", sim_matrix);
        Ok(())
    }

    #[test]
    fn ndarray_cosine_sim() -> anyhow::Result<()> {
        let embeds = Array1::range(0f32, 10., 1.).into_shape_clone((2, 5))?;
        let mut embeds_normed = embeds.clone();
        normalize_l2(&mut embeds_normed);
        assert_eq!(embeds_normed.shape(), [2, 5]);
        println!("{:8.16}", embeds_normed);

        let mut embeds = embeds.to_owned();
        let sim_matrix = similarity_matrix(&mut embeds);
        println!("{:8.16}", sim_matrix);
        assert_eq!(sim_matrix.shape(), [2, 2]);
        println!("{:8.16}", sim_matrix);
        Ok(())
    }

    #[test]
    fn test_f32_close() {
        let a = 1.0;
        let b = 1.0001;
        let r_tol = 0.001;
        assert!(f32_close(a, b, r_tol));

        let a = 1.0;
        let b = 1.1;
        let r_tol = 0.001;
        assert!(!f32_close(a, b, r_tol));

        let a = -0.08510615;
        let b = -0.08510614;
        let r_tol = 1e-6;
        assert!(f32_close(a, b, r_tol));
    }

    #[test]
    fn pair_cos_similarity() -> anyhow::Result<()> {
        let embed1_array = get_rand_arr1_f32(384, 0.0, 1.0)?;
        let embed2_array = get_rand_arr1_f32(384, 0.0, 1.0)?;

        let embed1 = embed1_array.flatten().to_vec();
        let embed2 = embed2_array.flatten().to_vec();
        let pair_sim = cos_similarity(&embed1, &embed2)?;
        println!("Pair sim: {:8.16}", pair_sim);
        let self_sim = cos_similarity(&embed1, &embed1)?;
        println!("Self sim: {:8.16}", self_sim);
        assert!(f32_close(1.0, self_sim, 1e-6));

        let mut embed1_array = embed1_array.into_shape_with_order((1, 384))?;
        let embed2_array = embed2_array.into_shape_with_order((1, 384))?;

        embed1_array.append(Axis(0), embed2_array.view())?;
        let sim_matrix = similarity_matrix(&mut embed1_array);
        let pair_sim_mx = sim_matrix.get((0, 1)).unwrap();
        println!("MX  sim: {:8.16}", pair_sim_mx);
        assert!(f32_close(pair_sim, *pair_sim_mx, 1e-6));

        Ok(())
    }

    #[test]
    fn read_safetensors() -> anyhow::Result<()> {
        //let tensor_file = "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/";
        //let tensor_file = "tests/test_data/superlinear_embeddings/bge-m3";
        let tensor_file = "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct/";
        let splits = load_splits_data(tensor_file)?;
        let mut embeddings = load_split_tensor(tensor_file, &splits[0])?;
        println!("{:8.16}", embeddings);
        let shape = embeddings.shape();
        let embeddings_flatten: Vec<f32> = embeddings.flatten().to_vec();
        let lx_scores = lexrank_array(&embeddings_flatten, shape[0], shape[1], None, 10000)?;
        println!("{:?}", lx_scores);
        Ok(())
    }

    #[test]
    fn superlinear_summary() -> anyhow::Result<()> {
        //let test_data_path = "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2";
        //let test_data_path = "tests/test_data/superlinear_embeddings/snowflake-arctic-embed-m-v1.5";
        //let test_data_path = "tests/test_data/superlinear_embeddings/bge-reranker-v2";
        let test_data_path = "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct";

        let splits_data = load_splits_data(&test_data_path)?;
        println!("{:?}", splits_data.len());
        for split_data in splits_data.iter() {
            println!(
                "################# {:?} #################",
                split_data.split_id
            );
            println!("{:?}", split_data.no_tokens);
            println!("{:?}", split_data.sentence_embeddings_file);
            let mut tensors = load_split_tensor(&test_data_path, &split_data)?;
            let shape = tensors.shape();
            let tensors_flatten: Vec<f32> = tensors.flatten().to_vec();

            let lx_rank = lexrank_array(&tensors_flatten, shape[0], shape[1], None, 10000)?;
            let lx_rank_str = lx_rank
                .iter()
                .map(|(idx, score)| format!("{:?}: {:.16}", idx, score))
                .collect::<Vec<String>>();
            println!("{:?}", lx_rank_str);
            let tops: Vec<_> = lx_rank.iter().take(2).collect();
            let summary: Vec<_> = tops
                .iter()
                .map(|(idx, _)| (idx, split_data.sentences.get(*idx).unwrap().clone()))
                .collect();
            println!("\nSummaries:");
            summary.iter().for_each(|(idx, sentence)| {
                println!("\t{:?} {:?}", idx, sentence);
            });
            println!("\n\n");
        }

        Ok(())
    }

    #[test]
    fn blas_superlinear_summary() -> anyhow::Result<()> {
        //let test_data_path = "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2";
        //let test_data_path = "tests/test_data/superlinear_embeddings/snowflake-arctic-embed-m-v1.5";
        //let test_data_path = "tests/test_data/superlinear_embeddings/bge-reranker-v2";
        let test_data_path = "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct";

        let splits_data = load_splits_data(&test_data_path)?;
        println!("{:?}", splits_data.len());
        for split_data in splits_data.iter() {
            println!(
                "################# {:?} #################",
                split_data.split_id
            );
            println!("{:?}", split_data.no_tokens);
            println!("{:?}", split_data.sentence_embeddings_file);
            let mut tensors = load_split_tensor(&test_data_path, &split_data)?;
            let shape = tensors.shape();
            let tensors_flatten: Vec<f32> = tensors.flatten().to_vec();

            let lx_rank = blas_lexrank_array(&tensors_flatten, shape[0], shape[1], None, 10000)?;
            let lx_rank_str = lx_rank
                .iter()
                .map(|(idx, score)| format!("{:?}: {:.16}", idx, score))
                .collect::<Vec<String>>();
            println!("{:?}", lx_rank_str);
            let tops: Vec<_> = lx_rank.iter().take(2).collect();
            let summary: Vec<_> = tops
                .iter()
                .map(|(idx, _)| (idx, split_data.sentences.get(*idx).unwrap().clone()))
                .collect();
            println!("\nSummaries:");
            summary.iter().for_each(|(idx, sentence)| {
                println!("\t{:?} {:?}", idx, sentence);
            });
            println!("\n\n");
        }

        Ok(())
    }
    #[test]
    fn normalize_l2_test() -> anyhow::Result<()> {
        let mut a = array![[1.0f32, 2., 3.], [4., 5., 6.],];
        normalize_l2(&mut a);
        println!("{:8.12}", a);
        Ok(())
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
    #[test]
    fn test_cosine_f32_matrix() {
        let rows = 3;
        let cols = 4;
        let embeds = rand_matrix(rows, cols);
        let result = blas_cosine_f32_matrix(&embeds, rows, cols);
        assert_eq!(result.len(), rows * rows);
        let sim_array = Array2::from_shape_vec((rows, rows), result).unwrap();
        println!("{:8.16}", sim_array);

        let mut simsimd_result = vec![0.0f32; rows * rows];
        for i in 0..rows {
            let a = &embeds[i * cols..(i + 1) * cols];
            for j in 0..rows {
                let b = &embeds[j * cols..(j + 1) * cols];
                let ss_cosine = 1.0 - f32::cosine(a, b).unwrap();
                simsimd_result[i * rows + j] = ss_cosine as f32;
                simsimd_result[j * rows + i] = ss_cosine as f32; // mirror
            }
        }
        let simsimd_array = Array2::from_shape_vec((rows, rows), simsimd_result).unwrap();
        println!("{:8.16}", simsimd_array);
    }
    const PARITY_DATASETS: [&str; 4] = [
        "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2",
        "tests/test_data/superlinear_embeddings/bge-reranker-v2",
        "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct",
        "tests/test_data/superlinear_embeddings/snowflake-arctic-embed-m-v1.5",
    ];

    /// Cosine similarities computed with f32 in different summation orders.
    const SIM_TOL: f32 = 1e-5;
    /// LexRank scores are about 1/n; the power method stops once an iteration moves the
    /// (un-normalised) eigenvector by less than 1e-5, so the two pipelines may stop one step apart.
    const SCORE_TOL: f32 = 1e-5;

    /// Every fixture split's similarity matrix is the same through ndarray, per-pair `sdot`
    /// and the `sgemm` Gram matrix.
    #[test]
    fn parity_similarity_matrix() -> anyhow::Result<()> {
        let (mut max_diff, mut max_diff_opt, mut splits) = (0f32, 0f32, 0);
        for data_path in PARITY_DATASETS {
            for split in load_splits_data(data_path)? {
                let (shape, embeddings) = load_split_vec(data_path, &split)?;
                let (rows, cols) = (shape[0], shape[1]);
                let nd = similarity_matrix(&mut array2_from_vec(&embeddings, &shape)?);
                let blas = blas_cosine_f32_matrix(&embeddings, rows, cols);
                let opt = blas_cosine_f32_matrix_opt(&embeddings, rows, cols);
                for i in 0..rows {
                    for j in 0..rows {
                        let expected = nd[[i, j]];
                        let d = (blas[i * rows + j] - expected).abs();
                        let d_opt = (opt[i * rows + j] - expected).abs();
                        assert!(
                            d <= SIM_TOL && d_opt <= SIM_TOL,
                            "{data_path} split {}: sim[{i}][{j}] ndarray {expected} sdot {} sgemm {}",
                            split.split_id,
                            blas[i * rows + j],
                            opt[i * rows + j]
                        );
                        max_diff = max_diff.max(d);
                        max_diff_opt = max_diff_opt.max(d_opt);
                    }
                }
                splits += 1;
            }
        }
        println!("{splits} splits, max |diff| sdot {max_diff:e}, sgemm {max_diff_opt:e}");
        Ok(())
    }

    /// `blas_lexrank_array` gives every sentence the score `lexrank_array` gives it, on every
    /// fixture split, with and without a threshold (continuous and discrete Markov matrix).
    #[test]
    fn parity_lexrank_scores() -> anyhow::Result<()> {
        let (mut max_diff, mut rankings) = (0f32, 0);
        for data_path in PARITY_DATASETS {
            for split in load_splits_data(data_path)? {
                let (shape, embeddings) = load_split_vec(data_path, &split)?;
                let (rows, cols) = (shape[0], shape[1]);
                for threshold in [None, Some(0.1), Some(0.3), Some(0.5)] {
                    let nd = lexrank_array(&embeddings, rows, cols, threshold, 10000)?;
                    let blas = blas_lexrank_array(&embeddings, rows, cols, threshold, 10000)?;
                    assert_eq!(nd.len(), rows);
                    assert_eq!(blas.len(), rows);

                    let mut nd_score = vec![f32::NAN; rows];
                    for &(idx, score) in &nd {
                        nd_score[idx] = score;
                    }
                    for &(idx, score) in &blas {
                        let d = (score - nd_score[idx]).abs();
                        assert!(
                            d <= SCORE_TOL,
                            "{data_path} split {} threshold {threshold:?}: sentence {idx} ndarray {} blas {score}",
                            split.split_id,
                            nd_score[idx]
                        );
                        max_diff = max_diff.max(d);
                    }

                    // Downstream keeps the top two; they may only differ where scores tie.
                    for k in 0..rows.min(2) {
                        assert!(
                            nd[k].0 == blas[k].0 || (nd[k].1 - blas[k].1).abs() <= SCORE_TOL,
                            "{data_path} split {} threshold {threshold:?}: rank {k} ndarray {:?} blas {:?}",
                            split.split_id,
                            nd[k],
                            blas[k]
                        );
                    }
                    rankings += 1;
                }
            }
        }
        println!("{rankings} rankings, max |score diff| {max_diff:e}");
        Ok(())
    }

    /// all-MiniLM-L6-v2 split 0 has negative cosine similarities, so a small relative threshold
    /// rescales to a negative absolute cut-off. That used to panic in `lexrank_array` and fail
    /// in `blas_lexrank_array`.
    #[test]
    fn threshold_with_negative_similarities() -> anyhow::Result<()> {
        let data_path = "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2";
        let splits = load_splits_data(data_path)?;
        let (shape, embeddings) = load_split_vec(data_path, &splits[0])?;
        let (rows, cols) = (shape[0], shape[1]);

        let sim = similarity_matrix(&mut array2_from_vec(&embeddings, &shape)?);
        let sim_min = sim.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            sim_min < -0.1,
            "fixture no longer has negative similarities: {sim_min}"
        );

        for threshold in [0.0, 0.05, 0.1] {
            let nd = lexrank_array(&embeddings, rows, cols, Some(threshold), 10000)?;
            let blas = blas_lexrank_array(&embeddings, rows, cols, Some(threshold), 10000)?;
            assert_eq!(nd.len(), rows);
            assert_eq!(blas.len(), rows);
            assert!(nd.iter().chain(&blas).all(|(_, score)| score.is_finite()));
        }
        Ok(())
    }

    /// A threshold outside [0, 1) is rejected with an error by both pipelines, not a panic.
    #[test]
    fn threshold_outside_unit_interval_is_an_error() -> anyhow::Result<()> {
        let data_path = "tests/test_data/superlinear_embeddings/gte-Qwen2-1.5B-instruct";
        let splits = load_splits_data(data_path)?;
        let (shape, embeddings) = load_split_vec(data_path, &splits[0])?;
        let (rows, cols) = (shape[0], shape[1]);

        for threshold in [-0.1, 1.0, 1.5, f32::NAN] {
            assert!(lexrank_array(&embeddings, rows, cols, Some(threshold), 10000).is_err());
            assert!(blas_lexrank_array(&embeddings, rows, cols, Some(threshold), 10000).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_softmax() -> anyhow::Result<()> {
        let rand_matrix = rand_matrix(4, 4);
        let arr = Array2::from_shape_vec((4, 4), rand_matrix.clone())?;
        println!("Before softmax:\n{:8.16}", arr);
        let nd_softmaxed = softmax(&arr)?;
        println!("After softmax:\n{:8.16}", nd_softmaxed);
        let blas_softmaxed = blas_softmax(&rand_matrix, 4, 4);
        let blas_softmaxed_arr = Array2::from_shape_vec((4, 4), blas_softmaxed)?;
        println!("After blas softmax:\n{:8.16}", blas_softmaxed_arr);
        Ok(())
    }
}
