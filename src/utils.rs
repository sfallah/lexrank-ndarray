use memmap2::MmapOptions;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitData {
    pub split_id: usize,
    pub no_tokens: usize,
    pub split_string: String,
    pub embeddings_tensors_file: String,
    pub embeddings_tensors_name: String,
    pub sentences: Vec<String>,
}

pub fn load_splits_data(data_path: &str) -> anyhow::Result<Vec<SplitData>> {
    let splits_data_path = format!("{}/splits_data.json", data_path);
    let splits_data = fs::read_to_string(splits_data_path)?;
    let splits_data: Vec<SplitData> = serde_json::from_str(&splits_data)?;
    Ok(splits_data)
}

pub fn load_split_vec(
    data_path: &str,
    split_data: &SplitData,
) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    let tensor_file = format!("{}/{}", data_path, split_data.embeddings_tensors_file);
    let file = File::open(tensor_file).expect("Failed to open file.");
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let (_, tensor) = SafeTensors::deserialize(&buffer)?
        .tensors()
        .into_iter()
        .next()
        .unwrap();
    let shape = tensor.shape();
    let ts_vec = buffer_to_vec(&tensor.data(), shape[0], shape[1])?;
    Ok((shape.to_vec(), ts_vec))
}
pub fn array2_from_vec(vec: &Vec<f32>, shape: &Vec<usize>) -> anyhow::Result<Array2<f32>> {
    if shape.len() != 2 {
        return Err(anyhow::anyhow!("Shape must have 2 dimensions"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let array: Array2<f32> = Array::from(vec.to_owned())
        .into_shape_clone((rows, cols))?
        .to_owned();
    Ok(array)
}

pub fn load_split_tensor(data_path: &str, split_data: &SplitData) -> anyhow::Result<Array2<f32>> {
    let (shape, vec) = load_split_vec(data_path, split_data)?;
    let array = array2_from_vec(&vec, &shape)?;
    Ok(array)
}

fn buffer_to_vec(buffer: &[u8], rows: usize, cols: usize) -> anyhow::Result<Vec<f32>> {
    // Ensure the buffer length matches the expected size
    let expected_len = rows * cols * size_of::<f32>();
    if buffer.len() != expected_len {
        return Err(anyhow::anyhow!(
            "Buffer size does not match the expected dimensions"
        ));
    }

    // Convert the buffer to a Vec<f32>
    let result: Vec<f32> = buffer
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(result)
}

pub fn get_rand_arr2_f32(
    m: usize,
    n: usize,
    mean: f32,
    std_dev: f32,
) -> anyhow::Result<Array2<f32>> {
    let rnd_arr = Array::random((m, n), Normal::new(mean, std_dev)?).into();
    Ok(rnd_arr)
}

pub fn get_rand_arr1_f32(n: usize, mean: f32, std_dev: f32) -> anyhow::Result<Array1<f32>> {
    let rnd_arr = Array::random(n, Normal::new(mean, std_dev)?).into();
    Ok(rnd_arr)
}
