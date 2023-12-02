# Assignment 5 Question 2 Solution

import time
import copy
import random
import argparse
import numpy as np
import pandas as pd
import random as rand
from tqdm import tqdm
from numpy.linalg import multi_dot
from itertools import combinations
from matplotlib import pyplot as plt
from ringbuf import RingBuffer

# Configure prints
np.set_printoptions(precision=0, suppress=True)

class TabuSearch: 

    def __init__(self, n=20, tenure=10, debug=False, seed=0, max_iter=10):
        
        # Debugging mode
        self.debug = debug

        # Number of sites & objects
        self.n = n

        # Best solution - assume 
        self.best_solution = [0] * n
        self.best_solution_cost = float('inf')

        # Store the best possible solution from that iteration
        self.current_solution = [0] * n

        # Tabu tenure 
        self.tabu_tenure = tenure

        # tabu list - stores cache of tenure lenght of recent swaps 
        self.tabu_list = RingBuffer(max_size=tenure) # use circular buffer for tabu list
        
        # ecodes the tabu information, frenquecy and recent swaps
        self.tabu_freq = np.zeros((self.n, self.n))

        # Use stopping criteria of 10 iterations
        self.max_iter = max_iter

        # Seed to initialize inital solution
        self.seed = seed

        # Keep track of best solution cost over time to visualize the changes
        self.costs = []

    def __init_solution(self):
        rand.seed(self.seed)
        s0 = list(range(0, self.n))
        rand.shuffle(s0)
        cost = self.objective_func(s0)

        print(f"Initial Solution: {s0} | Len[{len(s0)}] | Cost: {cost}")

        # Set best solution as initial
        self.best_solution = list(s0)
        self.best_solution_cost = cost

    def __gen_neighborhood(self, l, r=2):
        return list(combinations(l, r=r))

    def __swap(self, l, pos1, pos2):
        l[pos1], l[pos2] = l[pos2], l[pos1]
        return l

    def load_data(self, dist_csv="./data/distance.csv", flow_csv="./data/flow.csv"):
        
        pd_dist = pd.read_csv(dist_csv, index_col=False, header=None)
        pd_flow = pd.read_csv(flow_csv, index_col=False, header=None)

        self.distances = pd_dist.to_numpy()
        self.flows = pd_flow.to_numpy()

        # Get inital solution after data is loaded
        self.__init_solution()

        # self.debug and print(f"Distances: \n{self.distances} \n Flows: \n{self.flows}")

    def objective_func(self, site_vector):

        # N x N position matrix 
        position_matrix = np.zeros((self.n, self.n))
        
        # populate position matrix
        for idx, obj in enumerate(site_vector):

            # rows = site - index from 0
            # col = object - index from 0
            # a 1 at [a][b] indicates object b is connected to site a
            position_matrix[idx][obj] = 1

        # matrix multiplication (dot products)
        m2 = multi_dot([position_matrix, self.distances, position_matrix.T])

        # elementwise multiplication
        m3 = np.multiply(self.flows, m2)

        # retrieve cost
        cost = np.sum(m3)

        return cost

    def search(self, apsiration=False, sample_neighbors=False, neighborhood_frac=0.5, frequency=False):

        iterations = 0
        self.current_solution = copy.deepcopy(self.best_solution)
        while iterations < self.max_iter:

            # Track costs over each iteration
            self.costs.append([iterations, self.best_solution_cost])

            # generate candidate list
            candidate_swaps = self.__gen_neighborhood(self.current_solution, r=2)
            if sample_neighbors:
                # Only use a random portion of the neighborhood - change each iteration
                candidate_swaps = random.sample(candidate_swaps, round(len(candidate_swaps)*neighborhood_frac))

            prev_best_sol = [0]*self.n
            prev_best_cost = 9999
            prev_best_swap = (0,0)
            for swap in candidate_swaps:
                
                # get potential solution as a site vector
                ref = copy.deepcopy(self.current_solution)
                pot_sol = self.__swap(ref, swap[0], swap[1])

                # Check if in tabu list
                inTabu = self.tabu_list.contains(swap)

                if frequency:
                    cost = self.objective_func(pot_sol) + self.tabu_freq[swap[0]][swap[1]]
                else:
                    cost = self.objective_func(pot_sol)

                # Current Solution
                curr_solution = pot_sol
                curr_solution_cost = cost
                curr_sol_swap = swap

                # check if potenital solution is in tabu list
                if (not inTabu) and (curr_solution_cost < prev_best_cost):

                    prev_best_swap = copy.deepcopy(curr_sol_swap)
                    prev_best_sol = copy.deepcopy(curr_solution)
                    prev_best_cost = copy.deepcopy(curr_solution_cost)
                
                elif apsiration and (curr_solution_cost < prev_best_cost):

                    prev_best_swap = copy.deepcopy(curr_sol_swap)
                    prev_best_sol = copy.deepcopy(curr_solution)
                    prev_best_cost = copy.deepcopy(curr_solution_cost)

            self.debug and print(f"Iter[{iterations}]\nCurrent Sol: {prev_best_sol} | Cost: {prev_best_cost} | Swap {prev_best_swap}")
            self.debug and print(f"Best Sol: {self.best_solution} | Cost: {self.best_solution_cost}")
            self.debug and print(f"Tabu List: {self.tabu_list.buffer}")
            self.current_solution = copy.deepcopy(prev_best_sol)

            if prev_best_cost < self.best_solution_cost:
                self.debug and print(f">>> New Best Solution: {prev_best_swap}")
                self.best_solution = copy.deepcopy(prev_best_sol)
                self.best_solution_cost = copy.deepcopy(prev_best_cost)

            # Add to tabu list
            self.tabu_list.enqueue(prev_best_swap)

            # Track tabu frequency
            self.tabu_freq[prev_best_swap[0]][prev_best_swap[1]] = self.tabu_freq[prev_best_swap[0]][prev_best_swap[1]] + 1
            self.debug and print(f"Tabu Matrix\n{self.tabu_freq}")
            
            # Incerement
            iterations = iterations + 1

        print(f"\nBest Sol: {self.best_solution} | Cost: {self.best_solution_cost}")

    def plot_costs(self, showfig=True, savefig=False, fileName='QP'):

        fig = plt.figure(figsize=(16,9))

        path = np.array(self.costs)
        xs, ys = path.T
        plt.plot(xs, ys)

        plt.title("Tabu Search Costs vs Iteration")
        plt.xlabel('iteration', labelpad=20)
        plt.ylabel('Best Solution Cost', labelpad=20)

        if showfig:
            plt.show()

        if savefig:
            plt.savefig(f'figures/tabu_search_costs_{fileName}_{self.tabu_tenure + self.seed + self.n}.png')

def test():
    ts = time.time()

    # Testing data 
    test_tabu = TabuSearch(n=3, tenure=3, debug=True)
    test_tabu.load_data(dist_csv="./data/test_dist.csv", flow_csv="./data/test_flow.csv")

    test_c = test_tabu.objective_func([2, 1, 0])
    print(f"test cost: {test_c}")

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

def test_initial_seeds():

    for i in range(20):
        print(f"\nSeed={i}")
        tabu = TabuSearch(debug=False, seed=i, max_iter=400)
        tabu.load_data()
        tabu.search()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--full", 
        action='store_true',
        help='Full set of tests')
    parser.add_argument(
    "-s",
    "--single", 
    action='store_true',
    help='Run on one instance')

    return parser.parse_args()

def main():

    args = get_args()

    ts = time.time()

    # Test effect of random seeds
    # test_initial_seeds()

    if args.full:
        # set of seeds for different initial solutions
        seeds = [0, 2, 6]
        max_iterations = 400
        for seed in seeds:

            print(f"\n========================\nSeed[{seed}]\n========================\n")

            # Smaller Tenure
            print(f"\nInitial Seed:")
            tabu = TabuSearch(seed=seed, max_iter=max_iterations, tenure=10)
            tabu.load_data()
            tabu.search()

            # Smaller Tenure
            print(f"\nSmaller Tenure:")
            tabu = TabuSearch(seed=seed, max_iter=max_iterations, tenure=5)
            tabu.load_data()
            tabu.search()

            # Larger Tenure
            print(f"\nLarger Tenure:")
            tabu = TabuSearch(seed=seed, max_iter=max_iterations, tenure=20)
            tabu.load_data()
            tabu.search()

            # Aspiration added
            print(f"\nAspiration Added:")
            tabu = TabuSearch(seed=seed, max_iter=max_iterations, tenure=20)
            tabu.load_data()
            tabu.search(apsiration=True)

            # Smaller Neighborhood
            print(f"\nUsing 50% of Neighborhood:")
            tabu = TabuSearch(seed=seed, max_iter=max_iterations, tenure=20)
            tabu.load_data()
            tabu.search(sample_neighbors=True, neighborhood_frac=0.5, apsiration=True)

            # Frequency Tabu
            print(f"\nFrequency Tabu")
            tabu = TabuSearch(seed=seed, max_iter=max_iterations, tenure=20)
            tabu.load_data()
            tabu.search(frequency=True, sample_neighbors=True, neighborhood_frac=0.5, apsiration=True)

    elif args.single:
        print(f"\nSingle Instance")
        tabu = TabuSearch(seed=12, max_iter=250, tenure=10, debug=True)
        tabu.load_data()
        tabu.search()
        tabu.plot_costs(showfig=True)

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()