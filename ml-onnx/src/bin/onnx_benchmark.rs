use anyhow::Result;
use ort::{value::TensorRef};
use ndarray::{Array4};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::rng;
use indicatif::ProgressBar;
use image::{ImageBuffer, RgbImage};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use glob::glob;
use std::fs;
use csv::Writer;

// Constants
const TEST_IMAGES_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/data/test";
const ONNX_MODEL_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/models/RegNet_tuned.onnx";

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

fn main() -> Result<()> {

    println!("> Running ONNX Benchmark");

    let mut model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(8)?
    .commit_from_file(ONNX_MODEL_PATH)?;
    println!("ONNX model loaded successfully from: {}", ONNX_MODEL_PATH);

    // Load images
    let images_path_abs = fs::canonicalize(TEST_IMAGES_PATH).expect("Invalid path");

    // Build glob pattern: /abs/path/*.jpg
    let pattern = format!("{}/*.jpg", images_path_abs.display());

    // Collect matching paths
    let image_paths: Vec<String> = glob(&pattern)
        .expect("Failed to read glob pattern")
        .filter_map(Result::ok)               // unwrap Ok paths
        .filter(|p| p.exists())               // only keep existing files
        .map(|p| p.to_string_lossy().into())  // convert to String
        .collect();

    if image_paths.is_empty() {
        println!("No test images found! Please update the test_images list with actual image paths.");
    }

    let num_runs = 10;
    println!("\nBenchmarking batch of {} images with {} runs...", image_paths.len(), num_runs);

    let mut all_results: Vec<HashMap<String, String>> = Vec::new();
    let mut onnx_times: Vec<f64> = Vec::new();
    let mut rng = rng();

    // ---- Preprocess images ----
    let mut preprocessed_images: HashMap<String, Array4<f32>> = HashMap::new();
    let pre_pb = ProgressBar::new(image_paths.len() as u64);
    for path in &image_paths {

        let name = Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();

        let array_input = preprocess_image(path).unwrap();
        preprocessed_images.insert(name, array_input);
        pre_pb.inc(1);
    }

    // ---- Run benchmark loops ----
    println!("Images pre-processed");
    for loop_num in 0..num_runs {
        let mut keys: Vec<_> = preprocessed_images.keys().cloned().collect();
        keys.shuffle(&mut rng);

        let pb = ProgressBar::new(keys.len() as u64);
        pb.set_message(format!("Loop {}", loop_num + 1));

        for key in keys {
            let img_tensor = preprocessed_images.get(&key).unwrap();

            // Time ONNX inference
            let start = Instant::now();

            let outputs: SessionOutputs = model.run(ort::inputs!["x" => TensorRef::from_array_view(img_tensor)?])?;
            let predictions = outputs[0].try_extract_array::<f32>()?;
            let predictions_slice = predictions.as_slice().unwrap();
            let predicted_class_idx = argmax(predictions_slice);
            let probabilities = softmax(predictions_slice);
            let confidence = probabilities[predicted_class_idx];

            let elapsed = start.elapsed().as_secs_f64() * 1000.0; // convert to ms
            onnx_times.push(elapsed);

            // Fake results (replace with your softmax / argmax decoding)
            let rs_onnx_pred = predicted_class_idx;
            let rs_onnx_conf = confidence;

            let mut record = HashMap::new();
            record.insert("image".to_string(), key.clone());
            record.insert("rs_onnx_pred".to_string(), rs_onnx_pred.to_string());
            record.insert("rs_onnx_conf".to_string(), rs_onnx_conf.to_string());
            record.insert("rs_onnx_time_ms".to_string(), format!("{:.4}", elapsed));

            all_results.push(record);
            pb.inc(1);
        }
        pb.finish_with_message("done");
    }

    // ---- Summarize results ----
    let avg_onnx_time = onnx_times.iter().sum::<f64>() / onnx_times.len() as f64;
    let total_onnx_time: f64 = onnx_times.iter().sum();

    let mut batch_results = HashMap::new();
    batch_results.insert("num_images".to_string(), (image_paths.len() as f64) * (num_runs as f64));
    batch_results.insert("rs_onnx_avg_time_ms".to_string(), avg_onnx_time);
    batch_results.insert("rs_onnx_total_time_ms".to_string(), total_onnx_time);

    println!("{:#?}", batch_results);

    // Collect all unique keys = CSV headers
    let mut headers: Vec<String> = all_results
        .iter()
        .flat_map(|row| row.keys().cloned())
        .collect();
    headers.sort();
    headers.dedup();

    // Write to CSV
    let filename = "./data/benchmarks/benchmark_batch_rs.csv";
    let mut wtr = Writer::from_path(filename)?;
    wtr.write_record(&headers)?;

    for row in &all_results {
        let record: Vec<String> = headers
            .iter()
            .map(|h| row.get(h).cloned().unwrap_or_default())
            .collect();
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    println!("Saved results to {}", filename);
    Ok(())
}
