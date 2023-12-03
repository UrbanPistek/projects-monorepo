mod plot;

use plot::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Simulated Annealing...");

    #[allow(unused_assignments)]
    let mut res: Result<(), Box<dyn Error>> = plot_demo_2d();
    res = plot_opt_2d();

    return res;
}
