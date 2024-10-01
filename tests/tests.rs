mod tests {
    use ndarray::{Array, Array1};
    use ndarray_benches::{cosine_sim, get_rand_arr1_f32, get_rand_arr2_f32, linear_forward, normalize_l2};

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
    fn ndarray_rnd_cosine_sim() -> anyhow::Result<()> {
        let m = 25;
        let n = 384;
        let mean = 100.0;
        let std_dev = 15.0;
        let embeddings = get_rand_arr2_f32(m, n, mean, std_dev)?;
        let sim_matrix = cosine_sim(&embeddings)?;
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

        let sim_matrix = cosine_sim(&embeds)?;
        assert_eq!(sim_matrix.shape(), [2, 2]);
        println!("{:8.16}", sim_matrix);
        Ok(())
    }

}

