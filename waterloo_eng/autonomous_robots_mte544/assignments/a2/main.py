import pandas as pd
import numpy as np
from math import sqrt, trunc
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

def load_data(path="./data/data_RANSAC.csv"):
    
    df = pd.read_csv(path, index_col=False, header=None)
    return df.to_numpy()

def plot_raw_data_2d(data, showfig=True):

    fig = plt.figure(figsize=(16,9))
    dps = np.array(data)
    xs, ys = dps.T
    plt.scatter(xs, ys)

    plt.xlabel('x', labelpad=20)
    plt.ylabel('y', labelpad=20)

    if showfig:
        plt.show()

def plot_solution_data_2d(data, cx, cy, r, tol, k, title='Plot', savefig=False, filename_nodifier="best_solution"):

    # plt.figure(figsize=(16,9))
    fig, ax = plt.subplots(figsize=(16,9))
    
    dps = np.array(data)
    xs, ys = dps.T
    # alpha changes the transparency of the points
    plt.scatter(xs, ys, alpha=0.5)
    
    c = Circle((cx, cy), r, fill=False, color='r', linewidth=3.0, label="Estimate")
    cb1 = Circle((cx, cy), r+tol, fill=False, color='m', linestyle='--', linewidth=1.5, label="Upper Threshold")
    cb2 = Circle((cx, cy), r-tol, fill=False, color='g', linestyle='--', linewidth=1.5, label="Lower Threshold")
    ax.add_patch(c)
    ax.add_patch(cb1)
    ax.add_patch(cb2)

    plt.xlabel('x', labelpad=20)
    plt.ylabel('y', labelpad=20)
    plt.title(title)
    plt.legend()

    if savefig:
        kval = trunc(k*1000) # save decimal portion of the k value
        plt.savefig(f'figures/{filename_nodifier}_k{kval}.png')

class CircleRANSAC():

    def __init__(self, data, epsilon=0.1, max_iter=1000, debug=False):

        # maximun number of iterations to run the algorithm 
        self.max_iter = max_iter

        # data points to operate on 
        self.dataset = data

        # band size 
        self.band_tolerance = epsilon

        # total number of datapoints 
        self.dataset_len = len(self.dataset)

        # best solution
        # k is the fraction of points within the band tolerance 
        self.best_k = 0
        # stored as [(center), (edge)]
        self.best_points = []

        # for debugging
        self.debug = debug

    def __inside_outerband(self, center_x, center_y, rad, px, py):
        """
        Calculate whether point is within the upper bound.
        """

        p_rad = sqrt((px - center_x)**2 + (py - center_y)**2)
        if (p_rad <= rad):
            return True
        else:
            return False

    def __outside_innerband(self, center_x, center_y, rad, px, py):
        """
        Calculate whether point is outside the lower bound.
        """

        p_rad = sqrt((px - center_x)**2 + (py - center_y)**2)
        if (p_rad >= rad):
            return True
        else:
            return False

    def __frac_within_band(self, center_x, center_y, rad):
        """
        Calculate the fraction of total points within the model.
        """

        inliers = 0
        inner_rad = rad - self.band_tolerance
        outer_rad = rad + self.band_tolerance

        for point in self.dataset:
            is_inner = self.__inside_outerband(center_x, center_y, outer_rad, point[0], point[1])
            is_outer = self.__outside_innerband(center_x, center_y, inner_rad, point[0], point[1])

            if is_inner and is_outer:
                inliers += 1
        
        self.debug and print(f"==> Inliers: {inliers}")
        return (inliers/self.dataset_len)

    def fit(self):
        """
        Run RANSAC algorithm up to a max set of iterations and return the best result. 
        """

        # Store best so far parameters
        r = 0
        cx = 0
        cy = 0

        print(f"fitting model...")
        for iter in tqdm(range(self.max_iter)):

            # select 2 random points
            base_points = self.dataset[np.random.choice(self.dataset.shape[0], 2, replace=False)]
            self.debug and print(f"\niter[{iter}] Base: {base_points}")

            base_cx = base_points[0][0]
            base_cy = base_points[0][1]
            base_rx = base_points[1][0]
            base_ry = base_points[1][1]
            base_rad = sqrt((base_rx - base_cx)**2 + (base_ry - base_cy)**2)

            curr_k =  self.__frac_within_band(base_cx, base_cy, base_rad)
            
            # debugging
            if self.debug:
                print(f"Current k: {curr_k}, Best k: {self.best_k} | (cx, cy) = ({base_cx}, {base_cy}), r={base_rad}")
                plot_solution_data_2d(self.dataset, base_cx, base_cy, base_rad, self.band_tolerance)

            if curr_k > self.best_k: 
                self.best_k = curr_k
                self.best_points = base_points

                r = base_rad
                cx = base_cx
                cy = base_cy

        print(f"\n\nBest Solution: k={self.best_k}, center:{self.best_points[0]}, edge:{self.best_points[1]}")

        return cx, cy, r, self.best_k

def main():

    # load dataset
    data = load_data()

    # initialize object 
    iterations = 10000
    threshold = 0.7
    cRANSAC = CircleRANSAC(data, epsilon=threshold, max_iter=iterations)

    # perform fit and return parameters of the best fit
    cx, cy, r, k = cRANSAC.fit()

    # plot solution over dataset
    title = f"iter={iterations}, thres={threshold}, k={k}"
    filename = f"sol_i{iterations}_e{trunc(threshold*10)}"
    plot_solution_data_2d(data, cx, cy, r, cRANSAC.band_tolerance, k, title=title, savefig=True, filename_nodifier=filename)

if __name__ == "__main__":
    main()