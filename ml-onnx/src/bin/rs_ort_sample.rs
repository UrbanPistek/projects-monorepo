use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::{Array, Array4, Axis};
use serde::{Deserialize, Serialize};

use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::{
	value::TensorRef,
    value::Tensor,
    value::TensorRefMut
};

// Constants
const SAMPLE_MODEL_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/src/data/upsample.onnx";

fn argmax(array: &[f32]) -> usize {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}
fn main() -> Result<()> {
    
    let mut session = Session::builder()?.commit_from_file(SAMPLE_MODEL_PATH)?;
    let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
    let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
    println!("{:#?}", outputs);

    println!("ONNX sample model loaded successfully from: {}", SAMPLE_MODEL_PATH);

    // get the first output
    let output = &outputs[0];
    println!("{:#?}", output);

    // get an output by name
    let output = &outputs["Identity:0"];
    println!("{:#?}", output);
    
    let predictions = outputs["Identity:0"].try_extract_array::<f32>()?;
    println!("{:#?}", predictions);

    let predictions_slice = predictions.as_slice().unwrap();
    
    // // Find the predicted class
    let predicted_class_idx = argmax(predictions_slice);
    println!("{:#?}", predicted_class_idx);

    Ok(())
}
