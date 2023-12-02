import matplotlib.pyplot as plt
import numpy as np
from random import gauss
from math import sqrt, atan2, sqrt, asin

from state import State


def get_mean_state(particles: list[State]) -> State:
    return State(
        sum([state.x for state in particles])/len(particles),
        sum([state.y for state in particles])/len(particles),
        sum([state.theta for state in particles])/len(particles),
    )


def get_std_deviation(particles: list[State]) -> float:
    # returns the standard deviation of the state
    # assumes that the standard deviation of x, y, theta are independent
    sd_x = np.std([particle.x for particle in particles])
    sd_y = np.std([particle.y for particle in particles])
    sd_theta = np.std([particle.theta for particle in particles])

    return sqrt((sd_x**2 + sd_y**2 + sd_theta**2) / 3)

def euler_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


def plot(states: list[State], standard_deviations: list[float]) -> None:
    x = [state.x for state in states]
    y = [state.y for state in states]
    theta = [state.theta for state in states]

    plt.figure()
    plt.scatter(x[:-1], y[:-1], c='blue', label='Position at time < k')
    plt.scatter(x[-1], y[-1], c='red', label='Position at time = k')
    plt.legend()
    plt.title("Position over Time")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

    plt.figure()
    plt.scatter(range(len(x)), x)
    plt.title("x over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('x [m]')
    plt.show()

    plt.figure()
    plt.scatter(range(len(y)), y)
    plt.title("y over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('y [m]')
    plt.show()

    plt.figure()
    plt.scatter(range(len(theta)), theta)
    plt.title("Theta over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('theta [rad]')
    plt.show()

    plt.figure()
    plt.scatter(range(len(standard_deviations)), standard_deviations)
    plt.title("Standard Deviation over Time")
    plt.xlabel('Index of timestep')
    plt.ylabel('Standard Deviation [m]')
    plt.show()


def test_plot():
    states = []
    standard_deviations = []
    for i in range(100):
        states.append(State(
            gauss(0, 1),
            gauss(0, 1),
            gauss(0, 1),
        ))
        standard_deviations.append(gauss(0, 1))
    plot(states, standard_deviations)


def test_std_deviation():
    particles = []
    for i in range(1000):
        particles.append(State(
            gauss(0, 1),
            gauss(0, 1),
            gauss(0, 1),
        ))
    print(get_std_deviation(particles))


if __name__ == "__main__":
    test_std_deviation()
    test_plot()
