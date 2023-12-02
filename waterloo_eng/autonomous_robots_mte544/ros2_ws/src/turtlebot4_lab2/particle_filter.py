import numpy as np
from random import gauss

from state import State
from measurement import Measurement
from measurement_model import MeasurementModel
from sample_odom_model import SampleOdomModel


class ParticleFilter:
    def __init__(
        self, 
        num_particles: int,  # number of particles in the particle filter
        map_file: str,  # path to the file containing the map for the measurement model
        sd_p_hit: float,     # standard deviation of p_hit for the measurement model
        alpha_1:  float,     # alpha_1 for the motion model 
        alpha_2:  float,     # alpha_2 for the motion model
        alpha_3:  float,     # alpha_3 for the motion model
        alpha_4:  float,     # alpha_4 for the motion model
        ) -> None:

        self.num_particles = num_particles

        self.motion_model = SampleOdomModel(alpha_1, alpha_2, alpha_3, alpha_4)
        self.measurement_model = MeasurementModel(map_file, sd_p_hit)

    
    def initialize_particles(self, guess: State, standard_deviation: float) -> list[State]:
        initial_particles = []
        for m in range(self.num_particles):
            initial_particles.append(State(
                guess.x + gauss(0, standard_deviation),
                guess.y + gauss(0, standard_deviation),
                guess.theta + gauss(0, standard_deviation),
            ))
        return initial_particles

    def __call__(self, particles_prev: list[State], inputs: tuple[State, State], measurements: list[Measurement]) -> list[State]:
        weights = []
        predictions = []
        
        for m in range(self.num_particles):
            predicted_state = self.motion_model.calc_odom_sample(particles_prev[m], inputs)
            weight = self.measurement_model(measurements, predicted_state)
            predictions.append(predicted_state)
            weights.append(weight)

        # scale all the weights so they become probabilities (add up to 1)
        total_weight = sum(weights)
        probabilities = [weight/total_weight for weight in weights]

        # draw particles from predictions with probability proportional to weights
        indices = np.random.choice(range(self.num_particles), self.num_particles, replace=True, p=probabilities)
        particles_curr = [predictions[i] for i in indices]
        return particles_curr
