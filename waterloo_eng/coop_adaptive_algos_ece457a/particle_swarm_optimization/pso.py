import os
import copy
import numpy as np
import pandas as pd
import random as rand
from tqdm import tqdm
from math import sqrt, trunc
from matplotlib import pyplot as plt

class Particle():

    def __init__(self, pos: tuple, vel: tuple) -> None:
        self.position = pos
        self.velocity = vel

        # Initialize with float max
        self.best_position_val = 10.0**25
        self.global_best_val = 10.0**25
        
        self.best_position = (0, 0)
        self.global_best = (0, 0)

# PSO: Particle Swarm Optimization
class PSO:

    def __init__(self, n_particles=10, w=0.792, c1=1.4944, c2=1.4944, max_iter=100, update_method='sync') -> None:

        # Number of particles
        self.n_particles = n_particles

        # Have a hard upper limit on iterations
        self.max_iter = max_iter

        # paramters
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Scaling down velocities 
        self.r1 = rand.random()
        self.r2 = rand.random()

        # method for updating global best, either sync or async
        self.update_method = update_method

        # Tracking metrics
        self.best_solution_val = 10.0**25 # best solution value so far   
        self.best_solution = (0, 0) # best solution so far 
        self.best_particle = None # track best particle

        self.best_solutions = [] # all best solutions at each iteration
        self.mean_solutions = [] # mean solutions at each iteration
        self.mean_vel = [] # track mean velocities
        self.mean_velocity = 0 # track current mean velocity
        self.mean = 0 # mean solutions at each iteration
        self.iteration = 0 # track iterations completed

        # Bounds on search space saved as a absolute value
        self.xbounds = 5
        self.ybounds = 5

        # Velocity max saturator
        self.vx_max = 5
        self.vy_max = 5

        # Initialize particles
        self.__init_particles()

        # check convergence critieria
        self.convergence = self.__check_convergence_criteria()
        print(f"convergence: {self.convergence}\n")

    def __init_particles(self) -> None:

        self.swarm = []
        for _ in range(self.n_particles):

            # generate random starting position
            xi = rand.randint(-self.xbounds, self.xbounds)
            yi = rand.randint(-self.ybounds, self.ybounds)

            # generate small random starting velocity
            vx = (rand.randint(-self.xbounds, self.xbounds))/100
            vy = (rand.randint(-self.ybounds, self.ybounds))/100

            particle = Particle((xi, yi), (vx, vy))
            self.swarm.append(particle)

    def __check_convergence_criteria(self) -> bool:
        c = (self.c1 + self.c2)/2
        return (self.w < 1) and (c > 0) and ((2*self.w - c + 2) > 0)

    def __calc_velocity(self, v: float, f: float, pbest: float) -> float:
        """
        Calculate velocity in a singular dimension.
        """
        return (self.w*v + self.c1*self.r1*(pbest*self.iteration - f) + self.c2*self.r2*(self.best_solution_val*self.iteration - f))

    def __objective_func(self, x: float, y: float) -> float:
        """
        Evaluation of six-hump camelback function. 
        """
        return (4 - 2.1*(x**2) + (x**4)/3)*(x**2) + x*y + (-4 + 4*(y**2))*(y**2)

    def __update_particles(self):

        for particle in self.swarm:

            # Get previous values
            px = particle.position[0]
            py = particle.position[1]
            pvx = particle.velocity[0]
            pvy = particle.velocity[1]
            pbest = particle.best_position_val

            # Calculate velocities and positions
            vx = self.__calc_velocity(pvx, px, pbest)
            vy = self.__calc_velocity(pvy, py, pbest)

            # Saturate velocities if needed
            if vx > self.vx_max:
                vx = np.sign(vx)*self.vx_max
            
            if vy > self.vy_max:
                vy = np.sign(vy)*self.vy_max

            # Add velocities to position
            x = px + vx
            y = py + vy
            
            # Update particle values
            particle.position = (x, y)
            particle.velocity = (vx, vy)

            # check for new pbest
            sol = self.__objective_func(x, y)
            if sol < pbest:
                particle.best_position = (x, y)
                particle.best_position_val = sol

            # determine global best
            # if ssync method selected
            if self.update_method == 'async':
                self.__update_global_best()

            particle.global_best = self.best_solution
            particle.global_best_val = self.best_solution_val

    def __update_global_best(self) -> None:

        # find best particle
        self.swarm.sort(key=lambda particle: particle.best_position_val, reverse=False)
        best_particle = copy.deepcopy(self.swarm[0])

        self.best_particle = best_particle
        self.best_solution_val = best_particle.best_position_val
        self.best_solution = best_particle.best_position

    def __update_metrics(self) -> None:

        # update best particle
        self.best_solutions.append(copy.deepcopy(self.best_particle.best_position_val))

        # mean value of all particles
        mean = np.mean([p.best_position_val for p in self.swarm])
        self.mean_solutions.append(mean)
        self.mean = mean

        # track mean of velcocities
        mean_vel = np.mean([sqrt((p.velocity[0])**2 + (p.velocity[0])**2) for p in self.swarm])
        self.mean_velocity = mean_vel
        self.mean_vel.append(mean_vel)

    def __save_results(self) -> None: 
        if not os.path.exists('results'):
            os.makedirs('results')

        # save
        cost = trunc(abs(self.best_solution_val*100000))

        if self.best_solution_val < 0:
            filename = f"results_best_neg_{cost}_method_{self.update_method}"
        else:
            filename = f"results_best_{cost}_method_{self.update_method}"

        # save results to text file
        with open(f'results/{filename}.txt', 'w') as f:
            f.write(f"Best Solution:\nvalue: {self.best_solution_val}, position: {self.best_solution}\n")
            f.write(f"\nMeans:\nmean velocity: {self.mean_velocity}, mean: {self.mean}\n")
            f.write(f"\nParams:\nw={self.w}\nc1={self.c1}\nc2={self.c2}\niterations={self.max_iter}\nmethod={self.update_method}\nparticles={self.n_particles}")

    def search(self, debug=False) -> None:
        """
        """

        # initialize global best based on initial particles
        self.__update_global_best()
        
        pbar = tqdm(total=self.max_iter) # progress bar
        while self.iteration < self.max_iter:

            # update particles
            self.__update_particles()

            # determine global best
            # if sync method selected
            if self.update_method == 'sync':
                self.__update_global_best()

            # update metrics
            self.__update_metrics()

            # for debugging
            if debug:
                print(f"[{self.iteration}] best: {self.best_solution_val}, mean: {self.mean}")
                print(f"swarm:\n{[p.best_position_val for p in self.swarm]}")
                print(f"mean velocity: {self.mean_velocity}")

            # update progress
            pbar.update(1)
            self.iteration += 1

        del pbar
        print(f"Complete: best: {self.best_solution_val}, position: {self.best_solution}")
        print(f"mean velocity: {self.mean_velocity}, mean: {self.mean}")

        # save results to file
        self.__save_results()

    def show_results(self, save=False) -> None:
        """
        """
        # get x-axis
        xs = list(range(self.iteration))

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(12,7), sharex=True)

        ax1.set_title(f"Best Solution | Best={self.best_solution_val}")
        ax1.plot(xs, self.best_solutions, c='r', label='best')
        ax1.set_ylabel("z")
        ax1.legend()

        ax2.set_title(f"Mean Solution")
        ax2.plot(xs, self.mean_solutions, c='c', label='mean')
        ax2.set_ylabel("z")
        ax2.legend()
        ax2.set_ylim( min(self.mean_solutions)-1, max(self.mean_solutions)+1)

        ax3.set_title(f"Mean Velocity")
        ax3.plot(xs, self.mean_vel, c='m', label='velocity')
        ax3.set_ylabel("velocity")
        ax3.legend()
        ax3.set_ylim( min(self.mean_vel)-1, max(self.mean_vel)+1)

        if save:
            cost = trunc(abs(self.best_solution_val*100000))

            if self.best_solution_val < 0:
                filename = f"results_best_neg_{cost}_method_{self.update_method}"
            else:
                filename = f"results_best_{cost}_method_{self.update_method}"
            
            plt.savefig(f'figures/{filename}.png')
        else:
            plt.show()