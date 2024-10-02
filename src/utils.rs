use std::fs;
use ndarray::{Array, Array2};
use safetensors::SafeTensors;
use memmap2::MmapOptions;
use std::fs::File;

    use serde::{Deserialize, Serialize};
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

pub fn load_split_tensor(
    data_path: &str,
    split_data: &SplitData,
) -> anyhow::Result<Array2<f32>> {
    let tensor_file = format!("{}/{}", data_path, split_data.embeddings_tensors_file);
    let file = File::open(tensor_file).expect("Failed to open file.");
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let (_, tensor) = SafeTensors::deserialize(&buffer)?.tensors().into_iter().next().unwrap();
    let shape = tensor.shape();
    let ts_vec = buffer_to_vec(&tensor.data(), shape[0], shape[1])?;
    let emebeds_flatten = ts_vec.iter().flatten().cloned().collect::<Vec<f32>>();
    let array:Array2<f32> = Array::from(emebeds_flatten).into_shape_clone((shape[0], shape[1]))?.to_owned();
    Ok(array)
}

fn buffer_to_vec(buffer: &[u8], rows: usize, cols: usize) -> anyhow::Result<Vec<Vec<f32>>> {
    // Ensure the buffer length matches the expected size
    let expected_len = rows * cols * size_of::<f32>();
    if buffer.len() != expected_len {
        return Err(anyhow::anyhow!("Buffer size does not match the expected dimensions"));
    }

    // Convert the buffer to a Vec<f32>
    let flat_vec: Vec<f32> = buffer
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    // Reshape the flat vector into Vec<Vec<f32>>
    let mut result = Vec::with_capacity(rows);
    for row in flat_vec.chunks_exact(cols) {
        result.push(row.to_vec());
    }

    Ok(result)
}


