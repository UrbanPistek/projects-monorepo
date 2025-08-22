use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::{Array, Array4, Axis};
use serde::{Deserialize, Serialize};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::{
	value::TensorRef
};

// Constants
const SAMPLE_IMAGE_PATH: &str = "../data/test/Image_7.jpg";
const ONNX_MODEL_PATH: &str = "../models/RegNet_tuned.onnx";
const LABELS_MAPPING_PATH: &str = "../models/labels_mapping.json";

// Struct to hold the label mappings
#[derive(Debug, Serialize, Deserialize)]
struct LabelMapping {
    class_to_int: HashMap<String, i32>,
    int_to_class: HashMap<String, String>,
}

// Image preprocessing functions
fn center_crop(image: &RgbImage, crop_size: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    
    let start_x = (width - crop_size) / 2;
    let start_y = (height - crop_size) / 2;
    
    let mut cropped = ImageBuffer::new(crop_size, crop_size);
    
    for y in 0..crop_size {
        for x in 0..crop_size {
            let src_x = start_x + x;
            let src_y = start_y + y;
            cropped.put_pixel(x, y, *image.get_pixel(src_x, src_y));
        }
    }
    
    cropped
}

fn preprocess_image(image_path: &str) -> Result<Array4<f32>> {
    // Load and convert image to RGB
    let image = image::open(image_path)?;
    let image = image.to_rgb8();
    
    // Resize to 256x256 (equivalent to transforms.Resize(256))
    let image = image::imageops::resize(&image, 256, 256, image::imageops::FilterType::Lanczos3);
    
    // Center crop to 224x224 (equivalent to transforms.CenterCrop(224))
    let image = center_crop(&image, 224);
    
    // Convert to ndarray and normalize
    let mut array = Array4::<f32>::zeros((1, 3, 224, 224));
    
    // ImageNet normalization values
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    
    for y in 0..224 {
        for x in 0..224 {
            let pixel = image.get_pixel(x, y);
            
            // Convert to [0, 1] range and apply ImageNet normalization
            for c in 0..3 {
                let normalized = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                array[[0, c, y as usize, x as usize]] = normalized;
            }
        }
    }
    
    Ok(array)
}

fn load_labels_mapping(path: &str) -> Result<HashMap<i32, String>> {
    let file_content = std::fs::read_to_string(path)?;
    let mapping: LabelMapping = serde_json::from_str(&file_content)?;
    
    // Convert string keys to integers for int_to_class mapping
    let mut int_to_class = HashMap::new();
    for (key, value) in mapping.int_to_class {
        let int_key: i32 = key.parse()?;
        int_to_class.insert(int_key, value);
    }
    
    Ok(int_to_class)
}

fn argmax(array: &[f32]) -> usize {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    
    exp_logits.iter().map(|&x| x / sum_exp).collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    // // Initialize ONNX Runtime environment
    // let environment = Environment::builder()
    //     .with_name("RegNet_Inference")
    //     .build()?;
    
    // // Create session with optimizations
    // let session = SessionBuilder::new(&environment)?
    //     .with_optimization_level(GraphOptimizationLevel::All)?
    //     .with_execution_providers([ExecutionProvider::CPU(Default::default())])?
    //     .with_model_from_file(ONNX_MODEL_PATH)?;
    
    let mut model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file(ONNX_MODEL_PATH)?;
    println!("ONNX model loaded successfully from: {}", ONNX_MODEL_PATH);
    
    // Load and preprocess the image
    println!("Loading and preprocessing image: {}", SAMPLE_IMAGE_PATH);
    let input_array = preprocess_image(SAMPLE_IMAGE_PATH).unwrap();
    
    // Run inference
    println!("Running inference...");

    // let tensor = input_array.into();
    // let outputs = model.run(ort::inputs!["data" => tensor]?)?;
    let outputs: SessionOutputs = model.run(ort::inputs!["images" => TensorRef::from_array_view(&input_array)?])?;
    let output = outputs["output0"].try_extract_array::<f32>()?.t().into_owned();
    
    // Extract predictions
    let predictions = output.view();
    let predictions_slice = predictions.as_slice().unwrap();
    
    // Find the predicted class
    let predicted_class_idx = argmax(predictions_slice);
    
    // Calculate probabilities using softmax
    let probabilities = softmax(predictions_slice);
    let confidence = probabilities[predicted_class_idx];
    
    // Load labels mapping
    let int_to_class = load_labels_mapping(LABELS_MAPPING_PATH)?;
    
    // Get the predicted class name
    let predicted_class_name = int_to_class
        .get(&(predicted_class_idx as i32))
        .unwrap_or(&"Unknown".to_string());
    
    // Print results
    println!("\n=== Inference Results ===");
    println!("Predicted class index: {}", predicted_class_idx);
    println!("Predicted class name: {}", predicted_class_name);
    println!("Confidence: {:.6}", confidence);
    
    // Print top 5 predictions
    let mut indexed_probs: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &prob)| (i, prob))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\n=== Top 5 Predictions ===");
    for (i, (class_idx, prob)) in indexed_probs.iter().take(5).enumerate() {
        let class_name = int_to_class
            .get(&(*class_idx as i32))
            .unwrap_or(&"Unknown".to_string());
        println!("{}. {} ({}): {:.6}", i + 1, class_name, class_idx, prob);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_center_crop() {
        // Create a test image
        let test_image = ImageBuffer::from_fn(300, 300, |x, y| {
            image::Rgb([
                (x % 256) as u8,
                (y % 256) as u8,
                ((x + y) % 256) as u8,
            ])
        });
        
        let cropped = center_crop(&test_image, 224);
        assert_eq!(cropped.dimensions(), (224, 224));
    }
    
    #[test]
    fn test_argmax() {
        let test_array = vec![0.1, 0.8, 0.1, 0.3, 0.2];
        assert_eq!(argmax(&test_array), 1);
    }
    
    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }
}