mod tests {
    use ndarray_benches::{cosine_sim, get_rand_arr1_f32, get_rand_arr2_f32, linear_forward};

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
    fn ndarray_cosine_sim() -> anyhow::Result<()> {
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

}

