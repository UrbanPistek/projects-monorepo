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

fn main() -> Result<()> {
    
    println!("Running ort rust sample");

    Ok(())
}
