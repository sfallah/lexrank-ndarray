mod tests {
    use ndarray_benches::{get_rand_arr1_f32, get_rand_arr2_f32, linear_forward};

    #[test]
    fn ndarray_matmul() -> anyhow::Result<()> {
        let m = 16;
        let n = 384;
        let k = 1536;
        let mean = 100.0;
        let std_dev = 15.0;
        let lhs = get_rand_arr2_f32(m, k, mean, std_dev)?;
        let rhs = get_rand_arr2_f32(k, n, mean, std_dev)?;
        let bias = get_rand_arr1_f32(n, mean, std_dev)?.into_dimensionality::<ndarray::Ix1>()?;
        let add = linear_forward(&lhs, &rhs, &bias)?;

        assert_eq!(add.shape(), [m, n]);
        println!("{:?}", add);
        Ok(())
    }
}
