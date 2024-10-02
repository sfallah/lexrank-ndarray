mod tests {
    use ndarray::{array, Array1, Array2};
    use ndarray_benches::{similarity_matrix, get_rand_arr1_f32, get_rand_arr2_f32, linear_forward, normalize_l2, lexrank, lexrank_ts};
    use ndarray_benches::utils::{load_split_tensor, load_splits_data};
    use ndarray::parallel::prelude::*;


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
        let embeddings = get_rand_arr2_f32(m, n, mean, std_dev)?;
        let sim_matrix = similarity_matrix(&embeddings)?;
        assert_eq!(sim_matrix.shape(), [m, m]);
        println!("{:8.16}", sim_matrix);
        Ok(())
    }

    #[test]
    fn ndarray_cosine_sim() -> anyhow::Result<()> {
        let embeds = Array1::range(0f32, 10., 1.).into_shape_clone((2, 5))?;
        let embeds_normed = normalize_l2(&embeds)?;
        assert_eq!(embeds_normed.shape(), [2, 5]);
        println!("{:8.16}", embeds_normed);

        let sim_matrix = similarity_matrix(&embeds)?;
        assert_eq!(sim_matrix.shape(), [2, 2]);
        println!("{:8.16}", sim_matrix);
        Ok(())
    }

    #[test]
    fn read_safetensors() -> anyhow::Result<()> {
        let tensor_file = "tests/test_data/superlinear_embeddings/MiniLM-L6-v2/";
        let splits = load_splits_data(tensor_file)?;
        let embeddings = load_split_tensor(tensor_file, &splits[0])?;
        println!("{:8.16}", embeddings);
        let lx_scores = lexrank_ts(&embeddings, None, 10000)?;
        println!("{:?}", lx_scores);
        Ok(())
    }
}

