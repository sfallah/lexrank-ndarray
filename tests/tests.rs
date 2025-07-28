#[cfg(test)]
pub mod tests {
    use lexrank_ndarray::testing::{
        f32_close, get_rand_arr1_f32, get_rand_arr2_f32, load_split_tensor, load_splits_data,
    };
    use lexrank_ndarray::{cos_similarity, lexrank_ts, norm, norm_wide, normalize_l2, similarity_matrix, transpose_wide, vec_to_wide, wide_extract, Wide, LANES};
    use ndarray::{array, Array1, Axis};

    #[test]
    fn test_tranpose_wide() -> anyhow::Result<()> {
        let wide = vec_to_wide(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0])?;
        let wide2 = vec_to_wide(&vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0, 0.0])?;
        let matrix = vec![wide, wide2];
        let transposed = transpose_wide(&matrix)?;
        for (i, row) in transposed.iter().enumerate() {
            println!("Row {}: {:?}", i, row);
        }
        Ok(())
    }

    #[test]
    fn test_vec_to_wide_empty() {
        let input: Vec<f32> = vec![];
        let result = vec_to_wide(&input).unwrap();
        assert!(result.is_empty(), "Expected empty output for empty input");
    }

    #[test]
    fn test_vec_to_wide_exact_chunk() {
        // Create vector with exactly 8 values.
        let input: Vec<f32> = (1..=LANES).map(|x| x as f32).collect();
        let result = vec_to_wide(&input).unwrap();
        assert_eq!(result.len(), 1, "Expected one wide chunk");
        for i in 0..LANES {
            assert_eq!(result[0].to_array()[i], input[i], "Mismatch at lane {}", i);
        }
    }

    #[test]
    fn test_vec_to_wide_inexact_chunk() {
        // Create vector with 10 values. Expect two wide chunks.
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let result = vec_to_wide(&input).unwrap();
        assert_eq!(result.len(), 2, "Expected two wide chunks");

        // Check first 8 values in the first wide chunk.
        for i in 0..LANES {
            assert_eq!(
                result[0].to_array()[i],
                input[i],
                "Mismatch at first chunk, lane {}",
                i
            );
        }

        // Check remaining values in the second wide chunk.
        // For lanes with no corresponding input, behavior depends on crate internals,
        // so we only check the lanes that should have been populated.
        for i in 0..2 {
            assert_eq!(
                result[1].to_array()[i],
                input[8 + i],
                "Mismatch at second chunk, lane {}",
                i
            );
        }
        for i in 2..LANES {
            assert_eq!(
                result[1].to_array()[i],
                0.0,
                "Expected zero padding at lane {}",
                i
            );
        }
    }

    #[test]
    fn test_norm_wide_empty() {
        let tensor: Vec<Wide> = vec![];
        let result = norm_wide(&tensor);
        assert_eq!(result, 0.0, "Expected error for empty tensor");
    }

    #[test]
    fn test_norm_wide_single() -> anyhow::Result<()> {
        let wides = vec_to_wide(&vec![3.0, 4.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0])?;
        let result = norm_wide(wides.as_slice());
        assert_eq!(result, 30f32.sqrt());
        Ok(())
    }

    #[test]
    fn test_norm_wide_more() -> anyhow::Result<()> {
        let arr1 = vec_to_wide(&vec![3.0, 4.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0])?;
        let arr2 = vec_to_wide(&vec![3.0, 4.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0])?;
        let mut combined = arr1.clone();
        combined.extend(arr2);
        let res1 = norm_wide(combined.as_slice());
        assert_eq!(res1, 60f32.sqrt());

        let vec = vec![
            3.0, 4.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, -1.0, 2.0,
        ];
        let arr3 = vec_to_wide(&vec)?;
        let res2 = norm_wide(&arr3);
        assert_eq!(res2, res1);
        Ok(())
    }

    #[test]
    fn test_norm_compare() -> anyhow::Result<()> {
        let vec1 = vec![3.0, 4.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0];
        let vec2 = vec![3.0, 4.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0];
        let mut combined_vec = vec1.clone();
        combined_vec.extend_from_slice(&vec2);

        let nd_arr1 = Array1::from(combined_vec.clone());
        let nd_normed1 = norm(&nd_arr1)?.into_scalar();
        assert_eq!(nd_normed1, 60f32.sqrt());
        let wd_arr_vec = vec_to_wide(&combined_vec)?;

        let wd_normed1 = norm_wide(&wd_arr_vec);
        assert_eq!(nd_normed1, wd_normed1);
        Ok(())
    }
    
    #[test]
    fn ndarray_rnd_cosine_sim() -> anyhow::Result<()> {
        let m = 25;
        let n = 384;
        let mean = 100.0;
        let std_dev = 15.0;
        let mut embeddings = get_rand_arr2_f32(m, n, mean, std_dev)?;
        let sim_matrix = similarity_matrix(&mut embeddings)?;
        assert_eq!(sim_matrix.shape(), [m, m]);
        println!("{:8.16}", sim_matrix);
        Ok(())
    }

    #[test]
    fn ndarray_cosine_sim() -> anyhow::Result<()> {
        let mut embeds = Array1::range(0f32, 10., 1.).into_shape_clone((2, 5))?;
        let embeds_normed = normalize_l2(&mut embeds)?;
        assert_eq!(embeds_normed.shape(), [2, 5]);
        println!("{:8.16}", embeds_normed);

        let sim_matrix = similarity_matrix(&mut embeds)?;
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
        let sim_matrix = similarity_matrix(&embed1_array)?;
        let pair_sim_mx = sim_matrix.get((0, 1)).unwrap();
        println!("MX  sim: {:8.16}", pair_sim_mx);
        assert!(f32_close(pair_sim, *pair_sim_mx, 1e-6));

        Ok(())
    }

    #[test]
    fn read_safetensors() -> anyhow::Result<()> {
        //let tensor_file = "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/";
        //let tensor_file = "tests/test_data/superlinear_embeddings/bge-m3";
        let tensor_file = "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2";
        let splits = load_splits_data(tensor_file)?;
        let mut embeddings = load_split_tensor(tensor_file, &splits[0])?;
        println!("{:8.16}", embeddings);
        let lx_scores = lexrank_ts(&mut embeddings, None, 10000)?;
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
            let tensor = load_split_tensor(&test_data_path, &split_data)?;
            let lx_rank = lexrank_ts(&tensor, None, 10000)?;
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
        let a = array![[1.0f32, 2., 3.], [4., 5., 6.],];
        let normed = normalize_l2(&a)?;
        println!("{:8.12}", normed);
        Ok(())
    }
}
