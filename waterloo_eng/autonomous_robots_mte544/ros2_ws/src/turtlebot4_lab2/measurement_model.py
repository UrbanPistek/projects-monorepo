import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin
from scipy.stats import norm

from state import State
from measurement import Measurement


class MeasurementModel:
    def __init__(self, file: str, standard_deviation: float) -> None:
        self.resolution = 0.03

        with open(file, 'rb') as fin:
            map = plt.imread(fin)

            # Show maze map
            # plt.imshow(map)
            # plt.show()

        # get the indices with obstacles, then multiply by the resolution (0.03) to convert to m
        self.obstacles = np.array(np.where(map == 0)) * self.resolution

        # precompute distances
        # use a resolution the same as the resolution of the map (0.03m)
        self.distances = np.zeros(map.shape)
        for i in range(self.distances.shape[0]):
            for j in range(self.distances.shape[1]):
                self.distances[i, j] = np.min(np.sqrt(np.square(i*self.resolution - self.obstacles[0]) + np.square(j*self.resolution - self.obstacles[1])))

        self.standard_deviation = standard_deviation  # standard deviation of p_hit
        self.p_hit = norm(0, self.standard_deviation)

    def __call__(self, measurements: list[Measurement], state: State) -> float:
        q = 1
        for measurement in measurements:
            if measurement.valid():
                # transform into global frame
                x = state.x + (measurement.x() * cos(state.theta)) - (measurement.y() * sin(state.theta))
                y = state.y + (measurement.y() * cos(state.theta)) - (measurement.x() * sin(state.theta))

                # get the distance to the closest obstacle
                dist = self.distances[round(x/self.resolution), round(y/self.resolution)]

                # update q, assuming that measurements are independent and ignoring p_max and p_rand
                q *= 5 * self.p_hit.pdf(dist)
        return q


if __name__ == "__main__":
    # ms = MeasurementModel("ros2_ws/src/turtlebot4_lab2/maze_data/map_maze_1.pgm", 0.01)
    ms = MeasurementModel("/home/urban/urban/uw/4a/mte544/autonomous_robots_mte544/ros2_ws/src/turtlebot4_lab2/maze_data/map_maze_1.pgm", 0.01)
    