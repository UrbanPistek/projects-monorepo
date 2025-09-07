use anyhow::Result;
use image::{ImageBuffer, RgbImage};
use ndarray::{Array4};

pub fn argmax(array: &[f32]) -> usize {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    
    exp_logits.iter().map(|&x| x / sum_exp).collect()
}

pub fn center_crop(image: &RgbImage, crop_size: u32) -> RgbImage {
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

pub fn preprocess_image(image_path: &str) -> Result<Array4<f32>> {
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
