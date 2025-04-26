#[cfg(test)]
pub mod tests {
    use lexrank_ndarray::testing::{
        f32_close, get_rand_arr1_f32, get_rand_arr2_f32, load_split_tensor, load_splits_data,
    };
    use lexrank_ndarray::{
        cos_similarity, cosine_arrays, lexrank_ts, maximal_marginal_relevance_ts, normalize_l2,
        similarity_matrix,
    };
    use ndarray::{array, s, Array1, Array2, Axis};

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
    fn cosine_array_sim() -> anyhow::Result<()> {
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
        let pair_sim_mx = cosine_arrays(&embed2_array, &embed1_array)?
            .flatten()
            .into_iter()
            .enumerate()
            .reduce(|a, b| if a.1 < b.1 { a } else { b })
            .unwrap();

        println!("MX  sim: {:8.16}", pair_sim_mx.1);
        assert!(f32_close(pair_sim, pair_sim_mx.1, 1e-6));

        Ok(())
    }

    #[test]
    fn read_safetensors() -> anyhow::Result<()> {
        //let tensor_file = "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/";
        //let tensor_file = "tests/test_data/superlinear_embeddings/bge-m3";
        let tensor_file = "tests/test_data/superlinear_embeddings/multilingual-e5-large-instruct/";
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
    fn max_marginal_comb_lexrank() -> anyhow::Result<()> {
        let test_data_path = "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2";

        let splits_data = load_splits_data(&test_data_path)?;
        println!("{:?}", splits_data.len());
        for split_data in splits_data.iter() {
            println!(
                "################# {:?} #################",
                split_data.split_id
            );
            println!("{:?}", split_data.no_tokens);
            println!("{:?}", split_data.sentence_embeddings_file);
            let mut tensor = load_split_tensor(&test_data_path, &split_data)?;

            let lx_rank = lexrank_ts(&tensor, None, 10000)?;
            let lx_first = lx_rank[0];
            println!("First sentence: {:?}", lx_first);
            let query_array = tensor
                .index_axis(Axis(0), lx_first.0)
                .into_owned()
                .insert_axis(Axis(0));
            tensor.remove_index(Axis(0), lx_first.0);
            let result_array = tensor.to_owned();

            let res = maximal_marginal_relevance_ts(&query_array, &result_array, None, None)?;

            let first_sentence = split_data.sentences.get(lx_first.0).unwrap();
            println!("First sentence: {:?}", first_sentence);
            println!("Result: {:?}", res);
            for sentence in res.iter() {
                let sentence_text = split_data.sentences.get(*sentence).unwrap();
                println!("Sentence: {:?}", sentence_text);
            }

            println!("===========  LexRanked sentences: ===========");
            for lex_ranked in lx_rank[1..4].iter() {
                let sentence = split_data.sentences.get(lex_ranked.0).unwrap();
                println!("Sentence: {:?}", sentence);
            }
        }

        Ok(())
    }

    #[test]
    fn max_marginal_simple() -> anyhow::Result<()> {
        let test_data_path = "tests/test_data/superlinear_embeddings/all-MiniLM-L6-v2";

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

            let query_array = tensor
                .index_axis(Axis(0), 0)
                .into_owned()
                .insert_axis(Axis(0));

            let result_array = tensor.slice_axis(Axis(0), (1..).into()).to_owned();

            let res = maximal_marginal_relevance_ts(&query_array, &result_array, None, None)?;

            let res = res.iter().map(|idx| *idx + 1).collect::<Vec<usize>>();

            let first_sentence = split_data.sentences.get(0).unwrap();
            println!("First sentence: {:?}", first_sentence);
            println!("Result: {:?}", res);
            for sentence in res.iter() {
                let sentence_text = split_data.sentences.get(*sentence).unwrap();
                println!("Sentence: {:?}", sentence_text);
            }
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
