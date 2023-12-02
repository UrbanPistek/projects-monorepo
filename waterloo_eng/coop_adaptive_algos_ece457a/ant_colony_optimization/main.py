# Assignment 7 Question 3 Solution
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

from aco import ACO

# Configure prints
np.set_printoptions(precision=3, suppress=True)

# enable interactice mode
plt.ion()

def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser.parse_args:
    parser.add_argument(
        "-m",
        "--minimal", 
        action='store_true',
        help='Run with minimal live visuals')
    parser.add_argument(
        "-f",
        "--full", 
        action='store_true',
        help='Run with full live visuals')
    parser.add_argument(
        "-d",
        "--debug", 
        action='store_true',
        help='Run with full live visuals and debugging')

    return parser.parse_args()

def main():    
    print("Ant Colony Optimization...\n")

    parser = argparse.ArgumentParser()
    args = get_args(parser)

    # configure live visuals
    live_visuals = "no"
    if args.minimal:
         live_visuals = "minimal"
    elif args.full:
         live_visuals = "full"

    ts = time.time()

    # Create ACO Instance
    aco = ACO(live_visuals=live_visuals, m=25, evaporation=0.95, a=1, b=1, max_iter=10000)

    # show distances grid
    if args.debug:
        aco.visualize_grid(show_dist=True, show_pheromone=False)

    # perform search
    aco.search()

    # save plot and results
    plt.ioff()
    aco.show_results(save=True)

    te = time.time()

    print(f"\nElapsed time: {te - ts}s")

if __name__ == "__main__":
    main()