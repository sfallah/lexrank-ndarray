mod tests {
    use ndarray::{array, Array1, Axis};
    use ndarray_benches::{similarity_matrix, get_rand_arr1_f32, get_rand_arr2_f32, linear_forward, normalize_l2, lexrank, lexrank_ts};
    use ndarray_benches::utils::{load_split_tensor, load_splits_data};
    use ndarray::parallel::prelude::*;
    use ndarray_rand::rand_distr::num_traits::Pow;

    #[test]
    fn ndarray_matmul() -> anyhow::Result<()> {
        let m = 16;
        let n = 384;
        let k = 1536;
        let mean = 100.0;
        let std_dev = 15.0;
        let lhs = get_rand_arr2_f32(m, k, mean, std_dev)?;
        let rhs = get_rand_arr2_f32(n, k, mean, std_dev)?;
        let bias = get_rand_arr1_f32(n, mean, std_dev)?;
        let add = linear_forward(&lhs, &rhs, &bias)?;

        assert_eq!(add.shape(), [m, n]);
        println!("{:8.4}", add);
        Ok(())
    }

    #[test]
    fn ndarray_parallel_test() -> anyhow::Result<()> {
        let a = array![
                [1.,2.,3.],
                [4.,5.,6.],
            ];

        a.flatten().to_owned().into_par_iter().for_each(|x| {
            println!("{:?}", x);
        });
        let min: f32 = *a.flatten().to_owned().into_par_iter().min_by(|arg0, other| f32::partial_cmp(*arg0, *other).unwrap()).unwrap();
        assert_eq!(min, 1.0);
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
        let test_data_path = "tests/test_data/superlinear_embeddings/MiniLM-L6-v2";
        //let test_data_path = "tests/test_data/superlinear_embeddings/multilingual-e5-large-instruct";
        //let test_data_path = "tests/test_data/superlinear_embeddings/bge-m3";

        let splits_data = load_splits_data(&test_data_path)?;
        println!("{:?}", splits_data.len());
        for split_data in splits_data.iter() {
            println!(
                "################# {:?} #################",
                split_data.split_id
            );
            println!("{:?}", split_data.no_tokens);
            println!("{:?}", split_data.embeddings_tensors_file);
            let tensor = load_split_tensor(&test_data_path, &split_data)?;
            let lx_rank = lexrank_ts(&tensor, Some(0.3), 10000)?;
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
    fn pow_test() -> anyhow::Result<()> {
        let a = array![
            [1.0f32, 2., 3.],
            [4., 5., 6.],
        ];
        let b = a.mapv(|x| x.pow(2.0));
        println!("{:8.3}", b);



        let c = a.pow2();
        println!("{:8.3}", c);
        let c_norm = c.sum_axis(Axis(1)).sqrt();
        println!("c_norm: {:8.7}", c_norm);


        let mut d = array![
            [1.0f32, 2., 3.],
            [4., 5., 6.],
        ];
        d.par_mapv_inplace(|x| x.pow(2.0f32));
        println!("{:8.3}", d);

        let d_norm = d.sum_axis(Axis(1)).sqrt();
        println!("d_norm: {:8.7}", d_norm);
        Ok(())
    }

    #[test]
    fn normalize_l2_test() -> anyhow::Result<()> {
        let a = array![
            [1.0f32, 2., 3.],
            [4., 5., 6.],
        ];
        let normed = normalize_l2(&a)?;
        println!("{:8.12}", normed);
        Ok(())
    }
}

