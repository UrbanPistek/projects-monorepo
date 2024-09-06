// Calculate the magnitude of a vector by summing the squares of its coordinates
// and taking the square root. Use the `sqrt()` method to calculate the square
// root, like `v.sqrt()`.
fn magnitude(v: &[f64;3]) -> f64 {
    let sum: f64 = v[0].powf(2.0) + v[1].powf(2.0) + v[2].powf(2.0);
    let mag: f64 = sum.sqrt();
    return mag;
}

// Normalize a vector by calculating its magnitude and dividing all of its
// coordinates by that magnitude.
fn normalize(v: &mut [f64;3]) {
    let mag: f64 = magnitude(v);
    v[0] = v[0] / mag;
    v[1] = v[1] / mag;
    v[2] = v[2] / mag;
}

fn main() {
    println!("Magnitude of a unit vector: {}", magnitude(&[0.0, 1.0, 0.0]));

    let mut v = [1.0, 2.0, 9.0];
    println!("Magnitude of {v:?}: {}", magnitude(&v));
    normalize(&mut v);
    println!("Magnitude of {v:?} after normalization: {}", magnitude(&v));
}
