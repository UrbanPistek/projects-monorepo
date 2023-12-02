# Assignment 8 Question 3 Solution
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

from pso import PSO

# Configure prints
np.set_printoptions(precision=3, suppress=True)

def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser.parse_args:
    parser.add_argument(
        "-d",
        "--debug", 
        action='store_true',
        help='Run with debugging')

    return parser.parse_args()

# Plot six-hump camelback function
def plot_search_space():
    """
    3D Plot of six-hump camelback function.
    """

    fig = plt.figure(figsize = (12,7))
    ax = plt.axes(projection='3d')

    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = (4 - 2.1*(X**2) + (X**4)/3)*(X**2) + X*Y + (-4 + 4*(Y**2))*(Y**2)
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis)
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    plt.title("Search Space")

    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()

    # 2D Full Contour Plot
    x = np.arange(-2.5, 2.5, 0.01)
    y = np.arange(-2.5, 2.5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = (4 - 2.1*(X**2) + (X**4)/3)*(X**2) + X*Y + (-4 + 4*(Y**2))*(Y**2)

    fig = plt.figure(figsize = (12,7))
    num_lines = 25 # number of lines to draw in contour
    c = plt.contourf(X, Y, Z, num_lines) 
    fig.colorbar(c)
    plt.xlabel('x', labelpad=20)
    plt.ylabel('y', labelpad=20)
    plt.title("Search Space Full Contour")
    plt.show()

    # 2D Contour Plot
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = (4 - 2.1*(X**2) + (X**4)/3)*(X**2) + X*Y + (-4 + 4*(Y**2))*(Y**2)

    fig = plt.figure(figsize = (12,7))
    num_lines = 25 # number of lines to draw in contour
    c = plt.contour(X, Y, Z, num_lines) 
    fig.colorbar(c)
    plt.xlabel('x', labelpad=20)
    plt.ylabel('y', labelpad=20)
    plt.title("Search Space Contour")
    plt.show()

def main():    
    print("Particle Swarm Optimization...\n")

    parser = argparse.ArgumentParser()
    args = get_args(parser)

    ts = time.time()

    # visualize search space
    # plot_search_space()

    swarm = PSO(n_particles=100, max_iter=25, update_method='async')
    swarm.search(debug=False)
    swarm.show_results(save=True)

    te = time.time()

    print(f"\nElapsed time: {te - ts}s")

if __name__ == "__main__":
    main()