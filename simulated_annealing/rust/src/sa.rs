use rand::Rng;

pub struct SimulatedAnnealing1D {

    // Alpha value for decrementing temperature
    pub alpha: f32,

    // Initial temperature value
    pub temperature: f32,

    // Starting point - 1D
    pub start_x: f32,

    // Maximun number of iterations
    pub max_iterations: i32,

    // Track search path - x,y values
    path: Vec<(f32, f32)>,

    // Track total number of iterations
    iterations: i32,

    // Random number generator
    rng: rand::rngs::ThreadRng,

    // Constraint x to a specific domain
    search_domain_x: (f32, f32),

    // Track best solution [x, y]
    best_solution: (f32, f32),

}

impl SimulatedAnnealing1D {
    
    // Constructor
    pub fn new(a: f32, it: f32, sx: f32, mi: i32) -> SimulatedAnnealing1D {
        SimulatedAnnealing1D {
            alpha: a,
            temperature: it, // set initial temperature
            start_x: sx,
            max_iterations: mi,
            path: Vec::new(),
            iterations: 0,
            rng: rand::thread_rng(),
            search_domain_x: (0.0, 1.2),
            best_solution: (0.0, 0.0),
        }
    }

    // objecttive function
    fn objective_function(&self, x: f32) -> f32 {
        // 1.6 * x * sin(18.0*x)
        return 1.6 * x * f32::sin(18.0*x);
    }

    // Acceptance probability
    fn acceptance_probability(&mut self) -> f32 {

        // Generate a random number between 0 and 1
        return self.rng.gen_range(0.0..1.0);
    }
    
    // Probability function
    fn probability_function(&self, delta: f32) -> f32 {
        return (-delta / self.temperature).exp();
    }

    // Neighborhood function
    fn neighborhood_function(&mut self, x: f32) -> f32 {

        // Increment random in steps of up to 0.1
        let mut new_x: f32 = x + self.rng.gen_range(-0.25..0.25);

        // Constrain x to search domain
        new_x = new_x.max(self.search_domain_x.0);
        new_x = new_x.min(self.search_domain_x.1);

        return new_x;
    }

    // Run search
    pub fn search(&mut self) {

        // Initialize start
        let mut current_x: f32 = self.start_x;
        let mut current_solution: f32 = self.objective_function(current_x);

        self.best_solution = (current_x, current_solution);
        self.path.push((current_x, current_solution));

        // Loop until max iterations
        while self.temperature > 0.0 && self.iterations < self.max_iterations {

            // Use to determine if to iterate to the next temperature
            let mut interate: bool = false;

            // Get current x & solution
            current_x = self.path.last().unwrap().0;
            current_solution = self.path.last().unwrap().1;

            // Generate new x
            let new_x: f32 = self.neighborhood_function(current_x);

            // Determine if new x is better than the best so far x
            let current_new_solution: f32 = self.objective_function(new_x);

            // Here we are trying to find the global minimun
            // If new solution is better than the best so far
            if current_new_solution < current_solution {
                
                // Update path
                self.path.push((current_x, current_new_solution));
                interate = true;

            // If not use probability_function to determine if to go there anyway
            } else {

                let delta: f32 = current_new_solution.abs() - current_solution.abs();
                let probability: f32 = self.probability_function(delta);
                let accept_p: f32 = self.acceptance_probability();

                // If accept_p is less than probability then go there anyway
                if accept_p < probability {

                    // Update path
                    self.path.push((current_x, current_new_solution));
                    interate = true;
                }
            }

            // Check if current solution is best so far
            if current_new_solution < self.best_solution.1 {
                self.best_solution = (new_x, current_new_solution);
            }

            // Check if temperature should decrease
            if interate {

                // Using geometric cooling schedule
                self.temperature = self.temperature * self.alpha;

                // Update iterations
                self.iterations += 1;
            }
        }
    }

    pub fn get_stats(&self) -> (f32, f32, i32) {
        return (self.best_solution.0, self.best_solution.1, self.iterations);
    }

}