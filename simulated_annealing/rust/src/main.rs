mod plot;
mod sa;

use plot::plot_opt_2d;
use sa::SimulatedAnnealing1D;
use std::time::Instant;
use std::env;

fn main() {

    // Get command-line arguments
    let args: Vec<String> = env::args().collect();
    let num_args = args.len();

    // Access specific arguments
    let mut alpha: f32 = 0.99;
    let mut initial_temperature: f32 = 0.99;
    let mut start_x: f32 = 0.0;
    let mut max_iterations: i32 = 1000;

    if num_args > 4 {
        alpha = args[1].parse::<f32>().unwrap();
        initial_temperature = args[2].parse::<f32>().unwrap();
        start_x = args[3].parse::<f32>().unwrap();
        max_iterations = args[4].parse::<i32>().unwrap();
    }

    let start_time: Instant = Instant::now();
    println!("Simulated Annealing...");

    // Initialize Simulated Annealing
    let mut sa: SimulatedAnnealing1D = SimulatedAnnealing1D::new(alpha, initial_temperature, start_x, max_iterations);
    sa.search();
    let res: (f32, f32, i32) = sa.get_stats();

    // Print results
    println!("Best solution: x={}, y={}", res.0, res.1);
    println!("Iterations: {}", res.2);

    // Stop measuring time
    let end_time: Instant = Instant::now();

    // Calculate the elapsed time
    let elapsed_time: std::time::Duration = end_time.duration_since(start_time);
    println!("Elapsed time: {} us", elapsed_time.as_micros());

    // Plot results
    let _out = plot_opt_2d(res.0, res.1);
}
