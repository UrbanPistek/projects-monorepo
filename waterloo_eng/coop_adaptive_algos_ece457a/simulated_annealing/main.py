# Assignment 4 Question 3 Solution

import time 
import math
import numpy as np
import argparse
import matplotlib as mpl
import random as rand
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot Easom Function
def plot_function():

    fig = plt.figure(figsize = (16,9))
    ax = plt.axes(projection='3d')

    # x = np.arange(-10, 10, 0.1)
    # y = np.arange(-10, 10, 0.1)
    x = np.arange(0, 5, 0.1)
    y = np.arange(0, 5, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = -np.cos(X)*np.cos(Y)*np.exp(-np.power((X - math.pi), 2) - np.power((Y - math.pi), 2))
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis)

    # Marker at global minimum
    # ax.text(math.pi, math.pi, -1, "min")
    # ax.plot([math.pi], [math.pi], [-1], markerfacecolor='m', markeredgecolor='m', marker='o', markersize=5, alpha=0.6)
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()

def easom_func(x1, x2):
    z = -np.cos(x1)*np.cos(x2)*np.exp(-np.power((x1 - math.pi), 2) - np.power((x2 - math.pi), 2))
    return z

class SatInt:
    def __init__(self, val, lo, hi):
        self.real, self.lo, self.hi = val, lo, hi

    def __add__(self, other):
        return min(self.real + other.real, self.hi)

    def __sub__(self, other):
        return max(self.real - other.real, self.lo)

class SimulatedAnnealing:

    def __init__(self) -> None:
        
        # Alpha value for decrementing temperature
        # Linear annealing schedule
        # self.alpha = 0.001
        # Geometric annealing schedule
        self.alpha = 0.95

        # Initial temperature
        self.t = 0.95

        # Initial starting point 
        self.start = (rand.randint(-100, 100), rand.randint(-100, 100))

        # Global min to date
        # [x1, x2, z]
        self.g_min = [0,0,0]

        # All points visited
        self.path = []

        # Track Iterations
        self.iter = 1

        # Hard limit on max iterations
        # Linear annealing schedule
        # self.max_iter = round((1/(self.alpha)))
        
        # Geometric annealing schedule
        self.max_iter = 10000

        print(f"Initial: Alpha={self.alpha}, t={self.t}, start={self.start}, max_iter={self.max_iter}")

    def __objective_func(self, x1, x2):
        z = -np.cos(x1)*np.cos(x2)*np.exp(-np.power((x1 - math.pi), 2) - np.power((x2 - math.pi), 2))
        return z

    def __acceptance_probability(self):
        return rand.random()

    def __probability_func(self, delta):
        return np.exp(-delta/self.t)
    
    def __neighborhood_func(self, x1, x2):

        x1_new = SatInt(x1, -100, 100)
        x2_new = SatInt(x2, -100, 100)

        x1_new = x1_new + rand.random()*rand.randint(-1, 1)
        x2_new = x1_new + rand.random()*rand.randint(-1, 1)

        return x1_new, x2_new

    def search(self, debug=False):

        # Start
        z0 = self.__objective_func(self.start[0], self.start[1])
        p0 = [self.start[0], self.start[1], z0]
        self.g_min = p0
        self.path.append(p0)

        # While temperature is above 0 and max_iter is not exceeded
        while(self.t > 0 and self.iter < self.max_iter):

            # Use to track succesful iteration
            iterate = False

            # Get current step
            x1 = self.path[-1:][0][0]
            x2 = self.path[-1:][0][1]
            z = self.path[-1:][0][2]

            # Generate step
            x1n, x2n = self.__neighborhood_func(x1, x2)

            # Evaluate step
            zn = self.__objective_func(x1n, x2n)

            # See if step is better
            delta = abs(z) - abs(zn)
            if (delta < 0):

                # Move to next state
                pn = [x1n, x2n, zn]
                self.g_min = pn
                self.path.append(pn)

                # Iterate
                iterate = True
            else:
                # Calculate probability 
                ft = self.__probability_func(delta)
                accept = self.__acceptance_probability()

                if accept < ft:
                    # Move to next state
                    pn = [x1n, x2n, zn]
                    self.path.append(pn)

                    # store min only if its better
                    if abs(zn) > abs(self.g_min[2]):
                        self.g_min = pn

                    # Iterate
                    iterate = True

            if iterate:

                # update temperature & iterations
                # Linear annealing schedule
                # self.t = self.t - self.alpha
                
                # Geometric annealing schedule
                self.t = self.t*self.alpha

                self.iter = self.iter + 1

            if debug:
                print(f"\nIteration[{self.iter}], t={self.t}, global_min={self.g_min}")
                print(f"Step [{self.path[-1:][0]}]")
        
        print(f"END: start={self.start}, min={self.g_min} path_len={len(self.path)}")

    def plot_path_2d(self, showfig=True, savefig=False):

        fig = plt.figure(figsize=(16,9))
        ax = plt.axis([-100, 100, -100, 100]) 

        path = np.array(self.path)
        xs, ys, zs = path.T
        line = plt.plot(xs, ys, 'k',label='ployline',lw=2,marker='x',mew=3)
        
        plt.text(math.pi,math.pi, 'min', c='m')
        plt.text(self.start[0],self.start[1], 'start', c='m')
        plt.text(self.path[-1:][0][0],self.path[-1:][0][1], 'end', c='m')

        plt.xlabel('x1', labelpad=20)
        plt.ylabel('x2', labelpad=20)

        if showfig:
            plt.show()

        if savefig:
            randval = rand.randint(0, 10000)
            plt.savefig(f'figures/2d_path_fig{randval}.png')

    def plot_path_3d(self, showfig=True, savefig=False):

        fig = plt.figure(figsize = (16,9))
        ax = fig.add_subplot(111, projection='3d')

        path = np.array(self.path)
        xs, ys, zs = path.T
        
        ax.plot(xs, ys, zs, label='path')
        ax.legend()

        # Marker at global minimum
        ax.text(math.pi, math.pi, -1.05, "min")
        ax.text(self.start[0],self.start[1], 0, 'start', c='m')
        ax.text(self.path[-1:][0][0],self.path[-1:][0][1], self.path[-1:][0][2], 'end', c='m')
        ax.plot([math.pi], [math.pi], [-1], markerfacecolor='m', markeredgecolor='m', marker='o', markersize=5, alpha=0.6)
        
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)

        if showfig:
            plt.show()

        if savefig:
            randval = rand.randint(0, 10000)
            plt.savefig(f'figures/3d_path_fig{randval}.png')

def test_function():

    for i in range(0, 10):
        val = easom_func(i, i)
        print(f"f({i}, {i}) = {val}")
    
    val = easom_func(math.pi, math.pi)
    print(f"f({math.pi}, {math.pi}) = {val}")

    val = easom_func(-50, 50)
    print(f"f({-50}, {50}) = {val}")

    val = easom_func(20, 20)
    print(f"f({20}, {20}) = {val}")
    
    plot_function()

def main():

    # Test behavour of function at certin values
    test_function()
    
    ts = time.time()

    # Sample run
    sa = SimulatedAnnealing()
    sa.search(debug=True)
    sa.plot_path_2d()
    sa.plot_path_3d()

    for i in range(0, 10):
        print(f"Iteration: [{i}]")

        sa = SimulatedAnnealing()
        sa.search(debug=False)
        # sa.plot_path_2d(showfig=False, savefig=True)
        sa.plot_path_3d(showfig=False, savefig=True)

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()
