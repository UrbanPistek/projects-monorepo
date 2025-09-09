use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::{value::TensorRef};
use std::time::Instant;

// Constants
const MODEL_NAME: &str = "RegNet_x_400mf";
const SAMPLE_IMAGE_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/data/test/Image_7.jpg";
const LABELS_MAPPING_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/models/labels_mapping.json";

// Struct to hold the label mappings
#[derive(Debug, Serialize, Deserialize)]
struct LabelMapping {
    class_to_int: HashMap<String, i32>,
    int_to_class: HashMap<String, String>,
}

// Util functions
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

fn main() -> Result<()> {

    // Start timer
    let start = Instant::now();
    println!("Running Inference on: {:?}", MODEL_NAME);

    let onnx_model_path: String = format!("/home/urban/urban/projects/projects-monorepo/ml-onnx/models/{}_tuned.onnx", MODEL_NAME);
    let mut model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file(onnx_model_path)?;
    
    // Load and preprocess the image
    let array_input = onnx_inference::common::preprocess_image(SAMPLE_IMAGE_PATH).unwrap();

    // Run inference
    let start_inf = Instant::now();
    let outputs: SessionOutputs = model.run(ort::inputs!["x" => TensorRef::from_array_view(&array_input)?])?;

    // Extract predictions into usable format
    let predictions = outputs[0].try_extract_array::<f32>()?;
    let predictions_slice = predictions.as_slice().unwrap();
    
    // Find the predicted class
    let predicted_class_idx = onnx_inference::common::argmax(predictions_slice);
    
    // Calculate probabilities using softmax
    let probabilities = onnx_inference::common::softmax(predictions_slice);
    let confidence = probabilities[predicted_class_idx];
    let duration_inf = start_inf.elapsed();
    println!("Inference done in: {:?}", duration_inf);
    
    // Load labels mapping
    let int_to_class = load_labels_mapping(LABELS_MAPPING_PATH)?;
    
    // Get the predicted class name
    let default = String::from("Unknown");
    let predicted_class_name = int_to_class
        .get(&(predicted_class_idx as i32))
        .unwrap_or(&default);
    
    // Print results
    println!("\n=== Inference Results ===");
    println!("Predicted class index: {}", predicted_class_idx);
    println!("Predicted class name: {}", predicted_class_name);
    println!("Confidence: {:.6}", confidence);

    let duration = start.elapsed();
    println!("Completed in: {:?}", duration);
    
    Ok(())
}
