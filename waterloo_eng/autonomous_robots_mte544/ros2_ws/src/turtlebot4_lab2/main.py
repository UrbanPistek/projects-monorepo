import numpy as np
import pandas as pd

from particle_filter import ParticleFilter
from state import State
from measurement import Measurement, load_measurements
import utils

# For testing
from sample_odom_model import test_sample_odom, test_load_odometry_data

# Data loading
from sample_odom_model import load_odometry_data

# run tests
def test():
    test_sample_odom()
    test_load_odometry_data()

def main():
    measurement_csv_path = "bagreader/scan_point1.csv"
    odometry_csv_path = "bagreader/odom_point1.csv"
    map_path = "./maze_data/map_maze_1.pgm"
    num_particles = 10
    standard_deviation_p_hit = 1
    alpha_1 = 0.1
    alpha_2 = 0.1
    alpha_3 = 0.1
    alpha_4 = 0.1
    initial_guess_x = 54*0.03
    initial_guess_y = 210*0.03
    initial_guess_theta = 225*3.14/180
    initial_guess_standard_deviation = 0.1

    measurements = load_measurements(measurement_csv_path)
    odometry_readings = load_odometry_data(odometry_csv_path)

    particle_filter = ParticleFilter(
        num_particles, 
        map_path, 
        standard_deviation_p_hit,
        alpha_1,
        alpha_2,
        alpha_3,
        alpha_4)

    initial_guess = State(initial_guess_x, initial_guess_y, initial_guess_theta)
    particles = particle_filter.initialize_particles(initial_guess, initial_guess_standard_deviation)

    mean_states = [utils.get_mean_state(particles)]
    standard_deviations = [utils.get_std_deviation(particles)]
    for measurement, odom in zip(measurements, odometry_readings):
        particles = particle_filter(particles, odom, measurement)
        mean_states.append(utils.get_mean_state(particles))
        standard_deviations.append(utils.get_std_deviation(particles))

    utils.plot(mean_states, standard_deviations)


if __name__ == "__main__":
    main()
