import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.stats import norm
from math import sqrt


def get_obstacles(map_path: str) -> np.ndarray:
    """
    Reads the map and returns a list of obstacles
    :param map_path: path to map
    :return: array of shape (#obstacles, 2)
    """
    resolution = 0.03
    bottom_x, bottom_y = -1.94, -8.63

    with open(map_path, 'rb') as fin:
        map = plt.imread(fin)

        # Show maze map
        # plt.figure()
        # plt.imshow(map)
        # plt.show()

    # get the indices with obstacles, then multiply by the resolution (0.03) to convert to m
    obstacles = np.array(np.where(map == 0)) * resolution + np.array([[bottom_y], [bottom_x]])
    return np.transpose(obstacles)


def visualize_obstacles(obstacles: np.ndarray, save_path=None) -> None:
    """ visualize the obstacles"""
    plt.figure()
    plt.scatter(obstacles[:, 0], obstacles[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Obstacles")
    if save_path is not None:
        plt. savefig(save_path + "/obstacles.png")
    plt.show()


def get_measurements(csv_path: str) -> (np.ndarray, np.ndarray):
    """
    Read a measurement csv and get the bearings and ranges
    :param csv_path: path to csv
    :return:
    - bearings: np array of shape (#measurements, 719)
    - ranges: np array of shape (#measurements, 719)
    """
    df = pd.read_csv(csv_path, header=None, index_col=False)
    theta_min = df.iloc[0, 2]  # rad
    theta_max = df.iloc[0, 3]  # rad
    theta_inc = df.iloc[0, 4]  # rad

    bearings = np.arange(theta_min, theta_max, theta_inc)
    bearings = np.tile(bearings, (df.shape[0], 1))  # repeat the bearings to have the same shape as the bearings

    ranges = df.iloc[:, 10:].to_numpy()

    return bearings, ranges


def visualize_measurements(bearings: np.ndarray, ranges: np.ndarray, save_path=None) -> None:
    """Visualize the measurements"""
    x = ranges * np.cos(bearings)
    y = ranges * np.sin(bearings)
    x_long = np.reshape(x, -1)
    y_long = np.reshape(y, -1)
    plt.figure()
    plt.scatter(x_long, y_long, c='blue', label="measurements")
    plt.scatter(0, 0, c='red', label="robot position")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Measurements relative to robot position")
    if save_path is not None:
        plt. savefig(save_path + "/measurements.png")
    plt.show()


def get_init_particles(
        guess_x: float,
        guess_y: float,
        guess_theta: float,
        num_particles: float,
        init_guess_std_deviation: float
) -> np.ndarray:  # shape is (num_particles, 3)
    noise = np.random.normal(0, init_guess_std_deviation, (num_particles, 3))
    guess = np.array([guess_x, guess_y, guess_theta])
    return noise + guess


def convert_measurement_to_global(bearings, ranges, particle):
    """

    :param bearings: np aray of shape (#valid_ranges,)
    :param ranges: np array of shape (#valid_ranges,)
    :param particle: np array of shape (3,)
    :return: np array of shape (#valid_ranges, 3)
    """
    x = particle[0] + ranges * np.cos(particle[2] + bearings)
    y = particle[1] + ranges * np.sin(particle[2] + bearings)
    return np.transpose(np.array([x, y]))


def get_only_valid_measurements(bearings: np.ndarray, ranges: np.ndarray):
    """ only keeps values where range is finite"""
    valid_range_indices = np.isfinite(ranges)
    bearings = bearings[valid_range_indices]
    ranges = ranges[valid_range_indices]
    return bearings, ranges


def visualize_measurements_on_map(measurements: np.ndarray, obstacles: np.ndarray, final_state, save_path=None):
    print(obstacles.shape, "obstacles shape")
    print(measurements.shape, "measurement shape")

    plt.figure()
    plt.scatter(obstacles[:, 0], obstacles[:, 1], c="blue", label="obstacles")
    plt.scatter(measurements[:, 0], measurements[:, 1], c='red', label="measurements")
    plt.scatter(final_state[0], final_state[1], c='green', label='robot position')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.title("Obstacles and measurements")
    if save_path is not None:
        plt. savefig(save_path + "/obstacles and measurements.png")
    plt.show()


def plot(states: list[np.ndarray], standard_deviations: list[float], save_path=None) -> None:
    states = np.array(states)
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]

    plt.figure()
    plt.scatter(x[:-1], y[:-1], c='blue', label='Position at time < k')
    plt.scatter(x[-1], y[-1], c='red', label='Position at time = k')
    plt.legend()
    plt.title("Position over Time")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    if save_path is not None:
        plt. savefig(save_path + "/position.png")
    plt.show()

    plt.figure()
    plt.scatter(range(x.shape[0]), x)
    plt.title("x over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('x [m]')
    if save_path is not None:
        plt. savefig(save_path + "/x.png")
    plt.show()

    plt.figure()
    plt.scatter(range(y.shape[0]), y)
    plt.title("y over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('y [m]')
    if save_path is not None:
        plt. savefig(save_path + "/y.png")
    plt.show()

    plt.figure()
    plt.scatter(range(theta.shape[0]), theta)
    plt.title("Theta over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('theta [rad]')
    if save_path is not None:
        plt. savefig(save_path + "/theta.png")
    plt.show()

    plt.figure()
    plt.scatter(range(len(standard_deviations)), standard_deviations)
    plt.title("Standard Deviation over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('Standard Deviation [m]')
    if save_path is not None:
        plt. savefig(save_path + "/std dev.png")
    plt.show()


def main():
    map_path = "./maze_data/map_maze_1.pgm"
    measurement_csv_path = ["bagreader/scan_point1.csv", "bagreader/scan_point5.csv"]
    save_paths = ["results/point1", "results/point5"]
    guess_x = [-2.5, -1]
    guess_y = [0, -1]
    guess_theta = [-3.14/4, -3.14/4]

    num_particles = 1000
    init_guess_standard_deviation = 1.0
    lidar_standard_deviation = 0.2

    for point in range(2):
        obstacles = get_obstacles(map_path)
        bearings_all, ranges_all = get_measurements(measurement_csv_path[point])
        particles = get_init_particles(guess_x[point], guess_y[point], guess_theta[point], num_particles, init_guess_standard_deviation)

        visualize_obstacles(obstacles, save_paths[point])
        visualize_measurements(bearings_all, ranges_all, save_paths[point])

        kdt = KDTree(obstacles)

        particles_mean = [np.mean(particles, axis=0)]
        particles_std_deviation = [np.linalg.norm(np.std(particles, axis=0)) / sqrt(3)]
        for i, (bearings, ranges) in enumerate(zip(bearings_all, ranges_all)):
            print(f"Calibrating at measurement {i}")
            bearings, ranges = get_only_valid_measurements(bearings, ranges)
            weights = []
            for particle in particles:
                measurements = convert_measurement_to_global(bearings, ranges, particle)
                distances = kdt.query(measurements, k=1)[0][:]
                weights.append(np.prod(np.exp(-(distances ** 2) / (2 * lidar_standard_deviation ** 2))))

            # scale all the weights so they become probabilities (add up to 1)
            total_weight = sum(weights)
            weights = [weight/total_weight for weight in weights]

            # draw particles from predictions with probability proportional to weights
            indices = np.random.choice(range(num_particles), num_particles, replace=True, p=weights)
            particles = [particles[i] for i in indices]

            mean = np.mean(particles, axis=0)
            std_deviation = np.linalg.norm(np.std(particles, axis=0)) / sqrt(3)

            particles_mean.append(mean)
            particles_std_deviation.append(std_deviation)

            print(f"After measurement {i}, Mean state: {mean}, std deviation: {std_deviation}")

            if std_deviation < 0.001:
                print("std deviation is low. Breaking")
                break

        plot(particles_mean, particles_std_deviation, save_paths[point])

        measurements = convert_measurement_to_global(bearings, ranges, particles_mean[-1])
        visualize_measurements_on_map(measurements, obstacles, particles_mean[-1], save_paths[point])


if __name__ == "__main__":
    main()
