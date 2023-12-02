import os
import copy
import numpy as np
import pandas as pd
import random as rand
from tqdm import tqdm
from math import sqrt, trunc
from matplotlib import pyplot as plt

class Edge:

    def __init__(self, dist, pheromone):

        self.distance = dist
        self.pheromone = pheromone

    def update(self, dph):
        """
            dph: Delta phermone, update increment
        """
        self.pheromone += dph

class Ant:

    def __init__(self, tabu_len=29) -> None:
        
        self.tabu_list = [-1] * tabu_len # initialize with -1 to avoid conflicts
        self.cumul_dist = 0 # cumulative dist

# ACO: Ant Colony Optimization
class ACO:

    def __init__(self, m=25, evaporation=0.95, a=1, b=1, data_path='data/tsp.csv', live_visuals='no', max_iter=100) -> None:

        # Number of ants
        self.m = m

        # List of ants
        self.colony = [Ant()]*m
        self.__init_ants()

        # How quickly pheromones decay
        self.evaporation = evaporation

        # parameters to balance search
        self.alpha = a # local
        self.beta = b # global

        # Have a hard upper limit on iterations
        self.max_iter = max_iter
        self.keep_searching = True # bool to prevent stagnation

        # Number of nodes
        self.n = 29   

        # Track grid of pheromones
        self.pheromones = np.zeros((self.n, self.n))

        # Tracking metrics
        self.best_solution = Ant() # best solution so far 
        self.best_solution_iteration = 0 # track on which iteration the best solution was found
        self.current_best_solution = Ant() # best solution at the current iteration
        self.current_mean = 0 # current iteration mean
        self.current_std_dev = 1 # current iteration std deviation
        self.best_solutions = [] # all solutions at each iteration
        self.iteration_means = [] # means at each time step
        self.iteration_std_dev = [] # std devs at each time step
        self.costs = [] # track best cost over time
        self.mavs = [] # moving average of the mean value
        self.iteration = 0 # track iterations completed

        # How far of a lookback window to monitor stagnation
        self.stag_lookback = round(self.max_iter/10)

        # Load and preprocesss the data 
        self.__preprocess_data(data_path)

        # configure amount of visuals
        self.live_visuals = live_visuals

        # configure visuals
        if self.live_visuals != "no":
            self.__init_live_visuals()

    def __init_ants(self) -> None:
        """
        Initialize colony with unique ant objects
        """
        for i in range(self.m):
            self.colony[i] = Ant()

    def __init_live_visuals(self) -> None:
        """
        Initialize visuals for live plotting
        """
        
        # Plotting functionality
        if self.live_visuals == "full":
            fig1, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12,7))

            # Pheromones plot
            self.p_fig = fig1
            self.p_ax = ax1
            self.p_ax.set_title("Live Pheromones & Metrics")
            img = self.p_ax.imshow(np.zeros((29, 29)), cmap = "viridis")
            self.p_fig.colorbar(img, cmap="viridis")

            # Metrics plot
            self.m_ax = ax2
            self.line_mean, = self.m_ax.plot([], [], c='c', label='mean')
            self.line_bs, = self.m_ax.plot([], [], c='m', label='best')
            self.m_ax.legend()

            # cache backgrounds
            self.background1 = self.p_fig.canvas.copy_from_bbox(self.p_ax.bbox) # cache background
        
        elif self.live_visuals == "minimal":

            fig1, ax2 = plt.subplots(figsize=(12,7))

            # Metrics plot
            self.p_fig = fig1
            self.m_ax = ax2
            self.m_ax.set_title("Live Pheromones & Metrics")
            self.line_mean, = self.m_ax.plot([], [], c='c', label='mean')
            self.line_bs, = self.m_ax.plot([], [], c='m', label='best')
            self.m_ax.legend()

            # cache backgrounds
            self.background1 = self.p_fig.canvas.copy_from_bbox(self.m_ax.bbox) # cache background

    def __preprocess_data(self, data_path: str) -> None:
        """
        Loads data and pre-computes all the distances.
        """

        df = pd.read_csv(data_path, index_col=False)
        df = df.drop('city', axis=1) # drop city column
        data = df.to_numpy()

        init_pheromone = rand.random() # assign a random pheromone value to all edges between 0-1
        init_edge = Edge(dist=1, pheromone=init_pheromone)# Initial edge

        # Create graph representation in matrix form
        self.grid = np.full((self.n, self.n), init_edge)

        # pre-compute all the distances
        for row in range(self.n):

            # Base node
            xa =  data[row][0]
            ya = data[row][1]
            
            for col in range(self.n): 

                # Awat node
                xb = data[col][0]
                yb = data[col][1]
                
                # Compute the eculidean distance and update the edge
                dist = self.__calc_distance(xa, ya, xb, yb)
                edge = Edge(dist=dist, pheromone=init_pheromone)

                # create matrix representation of weighted graph
                self.grid[row][col] = edge

    def __calc_distance(self, xa: float, ya: float, xb: float, yb: float) -> float:
        """
        Eculidean distance
        """
        return sqrt(pow((xa -xb), 2) + pow((ya -yb), 2))

    def __ant_traversal(self, i: float) -> None:
        """
        Have a singular ant traverse all nodes in the graph
        """
        
        # Select ant and start at a random node
        ant = self.colony[i]
        start = rand.randint(0, 28)
        ant.tabu_list[0] = start

        # Get list of all possible nodes to travel to
        all = list(range(29))
        current_node = start

        # Iterate over all nodes
        for idx in range(1, self.n): 
            
            # Get list of nodes not yet visited using the difference between 2 sets
            valid = list(set(all) - set(ant.tabu_list))
            pks = self.__transition_rule(valid, current_node)

            # choose next node 
            node = rand.choices(valid, weights=pks, k=1) # returns a list
            node = node[0]

            # update the ants parameters
            ant.tabu_list[idx] = node
            ant.cumul_dist += self.grid[current_node][node].distance

            # update current node
            current_node = node

    def __transition_rule(self, nodes: list[int], current_node: int) -> list[float]:
        """
        Determine probabilities of all availible nodes a ant can visit
        """

        # compute total sum of (pheromones/distance) of all availible nodes
        denom = []
        for node in nodes:
            edge = self.grid[current_node][node]
            frac = pow(edge.pheromone, self.alpha)/pow(edge.distance, self.beta)
            denom.append(frac)

        # determine individual probabilities for each node
        d = sum(denom)
        probabilities = [((pow(self.grid[current_node][i].pheromone, self.alpha)/pow(self.grid[current_node][i].distance, self.beta))/d) for i in nodes]

        return probabilities

    def __update_pheromones(self) -> None:
        """
        Update all pheromones, first by applying evaporation then incrementing the pheromone values based on 
        the current iterations best solution.
        """
        
        # evaporate pheromones
        for row in range(self.n):
            for col in range(self.n): 
                self.grid[row][col].pheromone *= self.evaporation

                # update pheromone grid
                self.pheromones[row][col] = self.grid[row][col].pheromone

        # get the current iterations best ant
        ant = self.current_best_solution

        # update relevent edges based on solution
        delta = 100/ant.cumul_dist
        for i in range(1, len(ant.tabu_list)):
            city = ant.tabu_list[i]
            prev_city = ant.tabu_list[i-1]

            # update pheromone value
            self.grid[city][prev_city].pheromone += delta
            self.grid[prev_city][city].pheromone += delta

    def __update_metrics(self) -> None:
        """
        Update metrics to track best solution , mean solution and monitor stagnation
        """

        # sort fitness scores inplace, best score in highest index
        self.colony.sort(key=lambda ant: ant.cumul_dist, reverse=False)
        
        best_ant = self.colony[0]
        mean = np.mean([ant.cumul_dist for ant in self.colony])
        sd = np.std([ant.cumul_dist for ant in self.colony])

        if best_ant.cumul_dist < self.current_best_solution.cumul_dist:
            self.best_solution = best_ant
            self.best_solution_iteration = copy.deepcopy(self.iteration)

        # track current best solution values
        self.current_best_solution = best_ant # best solution at the current iteration
        self.current_mean = mean # current iteration mean
        self.current_std_dev = sd # current iteration std deviation

        # Track historical progress
        self.best_solutions.append(best_ant)
        self.iteration_means.append(mean)
        self.iteration_std_dev.append(sd)
        self.costs.append(best_ant.cumul_dist)

        # moving average
        if len(self.iteration_means) > 25:
            mav = np.ma.average(self.iteration_means[-25:])
        else:
            mav = np.ma.average(self.iteration_means)
        self.mavs.append(mav)

        # determine rate of change to monitor stagnation
        if self.iteration > self.stag_lookback:
            slope_intercept = np.polyfit(list(range(self.stag_lookback)), self.mavs[-self.stag_lookback:], 1)

            if abs(slope_intercept[0]) < 0.0005:
                self.keep_searching = False

        # Increment the number of iterations
        self.iteration += 1

    def __live_visuals_update(self) -> None:
        """
        Display live updates of best and mean solution
        """

        self.p_fig.canvas.restore_region(self.background1)

        if self.live_visuals == "full":

            # pheromones plot
            self.p_ax.imshow(self.pheromones, cmap="viridis") 
            # solutions over time
            self.line_mean, = self.m_ax.plot(self.iteration_means, c='c', label='mean')
            self.line_bs, = self.m_ax.plot(self.costs, c='m', label='best')
            self.p_fig.canvas.blit(self.p_ax.bbox)
                
        elif self.live_visuals == "minimal":

            # solutions over time
            self.line_mean, = self.m_ax.plot(self.iteration_means, c='c', label='mean')
            self.line_bs, = self.m_ax.plot(self.costs, c='m', label='best')
            self.p_fig.canvas.blit(self.m_ax.bbox)

        self.p_fig.canvas.flush_events()

    def search(self) -> None:
        """
        Apply ant colony optimization with the specified parameters
        """

        iter = 0
        pbar = tqdm(total=self.max_iter) # progress bar

        # Using a upper max iterations limit while monitoring for stagnation
        while (iter < self.max_iter) and self.keep_searching:

            # Traverse for all ants
            for i in range(len(self.colony)):
                self.__ant_traversal(i)

            # Compute metrics from traversal
            self.__update_metrics()

            # Update pheromones
            self.__update_pheromones()

            # Reset colony 
            self.__init_ants()

            # Show visuals
            if self.live_visuals != "no":
                self.__live_visuals_update()

            pbar.update(1)
            iter += 1

        del pbar

        # Print out final solution 
        print(f"Best Solution: {self.best_solution.cumul_dist} | Found at i={self.best_solution_iteration}")
        print(f"Traversal: {self.best_solution.tabu_list}")

    def show_results(self, save=False) -> None:
        """
        Show historical values over all iterations once search is complete
        """

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(12,7), sharex=True)

        # Caclulate trendlines
        xs = list(range(self.iteration))
        z = np.polyfit(xs, self.costs, 1)
        p = np.poly1d(z)

        # Means & Best plot
        ax1.set_title(f"Mean and Best Solution | Best={trunc(self.best_solution.cumul_dist)}")
        ax1.plot(xs, self.iteration_means, c='c', label='mean')
        ax1.plot(xs, self.costs, c='m', label='best')
        ax1.set_ylabel("Costs")

        # add trendline
        ax1.plot(xs, p(xs), c='k',label='trend')
        ax1.legend()

        # Std dev plot
        ax2.set_title(f"Standard Deviation")
        ax2.plot(xs, self.iteration_std_dev, c='c', label='sd')
        ax2.set_ylabel("Costs std")
        ax2.legend()

        # moving average plot
        ax3.set_title(f"Solution Moving Average")
        ax3.plot(xs, self.mavs, c='r', label='moving average')
        ax3.set_ylabel("Mean")
        ax3.set_xlabel("Iteration")
        ax3.legend()

        # save plot and results
        if save:
            if not os.path.exists('figures'):
                os.makedirs('figures')

            if not os.path.exists('results'):
                os.makedirs('results')

            # save plot
            cost = trunc(self.best_solution.cumul_dist)
            filename = f"results_m{self.m}_i{self.max_iter}_p{trunc(self.evaporation*100)}_a{trunc(self.alpha*100)}_b{trunc(self.beta*100)}_{cost}"
            plt.savefig(f'figures/{filename}.png')

            # save results to text file
            with open(f'results/{filename}.txt', 'w') as f:
                f.write(f"Best Solution:\n{self.best_solution.cumul_dist}\n{self.best_solution.tabu_list}")
                f.write(f"\nParams:\nm={self.m}\nalpha={self.alpha}\nbeta={self.beta}\niterations={self.max_iter}\nevaporation={self.evaporation}")
        else:
            plt.show()

    def visualize_grid(self, show_dist=False, show_pheromone=True) -> None:
        """
        Visualize pheromone and distances grid
        """

        pheromones = np.zeros((self.n, self.n))
        distances = np.zeros((self.n, self.n))

        # extract values
        for i in range(self.n):
            for j in range(self.n):
                pheromones[i][j] = self.grid[i][j].pheromone
                distances[i][j] = self.grid[i][j].distance

        # visualize pheromones
        if show_pheromone:
            plt.figure()
            plt.axis([0, 28, 0, 28]) 
            img = plt.imshow(pheromones, cmap = "plasma") 
            plt.colorbar(img, cmap="plasma")
            plt.title("Pheromones")
            plt.show()

        # visualize distances
        if show_dist:
            plt.figure()
            plt.axis([0, 28, 0, 28]) 
            img = plt.imshow(distances, cmap = "inferno") 
            plt.colorbar(img, cmap="inferno")
            plt.title("Distances")
            plt.show()