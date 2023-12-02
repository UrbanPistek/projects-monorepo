# Assignment 1 Question 3 Solution

import os 
import math
import time 
import random as rand
import numpy as np
import argparse

'''
Note that the first row in the 2D list is the y = 0 row 
(i.e. bottom-most row in the maze figure). 
'1' indicates that the node is blocked, '0' indicates that it is free.
'''
MAZE = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]

LOCATIONS = {
    "start": {
        "x": 2,
        "y": 11
    }, 
    "end1": {
        "x": 23,
        "y": 19
    },
    "end2": {
        "x": 2,
        "y": 21
    }
}

""""
Moves:

0 - up
1 - down
2 - left
3 - right
"""
MOVE_ENCODER = {
    "0": {
        "x_inc": 0, 
        "y_inc": 1
    }, 
    "1": {
        "x_inc": 0, 
        "y_inc": -1
    }, 
    "2": {
        "x_inc": -1, 
        "y_inc": 0
    }, 
    "3": {
        "x_inc": 1, 
        "y_inc": 0
    }
}

X_BOUNDS = 24
Y_BOUNDS = 24

class Agent:

    def __init__(self, x0, y0, debug = False):

        # track state
        self.curr_x = x0
        self.curr_y = y0
        self.prev_x = x0 
        self.prev_y = y0

        # track path, include start
        self.path_cost = 1
        self.path = [(x0, y0)]

        # utilities
        self.debug = debug

        print(f"Agent Initialized: ({self.curr_x}, {self.curr_y })")

    def check_next_move(self, x, y):
        
        if (x > X_BOUNDS) or (x < 0):
            return False
        
        if (y > Y_BOUNDS) or  (y < 0):
            return False

        if MAZE[x][y] == 1:
            return False

        if (x == self.prev_x) and (y == self.prev_y):
            return False

        return True

    def check_goal(self):

        if (self.curr_x == LOCATIONS["end1"]["x"]) and (self.curr_y == LOCATIONS["end1"]["y"]):
            print("Found: END1")
            return True, 1

        if (self.curr_x == LOCATIONS["end2"]["x"]) and (self.curr_y == LOCATIONS["end2"]["y"]):
            print("Found: END2")
            return True, 2

        return False, 0

    def reset(self, x, y):

        self.curr_x = x
        self.curr_y = y
        self.prev_x = x 
        self.prev_y = y

        # track path, include start
        self.path_cost = 1
        self.path = [(x, y)]
    
    def random_move(self):
        """"
        Moves:

        0 - up
        1 - down
        2 - left
        3 - right
        """

        # seed
        x_next = self.curr_x
        y_next = self.curr_y
        valid = False

        while not valid:

            # Generate random number
            move = rand.randint(0, 3)

            x_next = self.curr_x + MOVE_ENCODER[str(move)]["x_inc"]
            y_next = self.curr_y + MOVE_ENCODER[str(move)]["y_inc"]

            valid = self.check_next_move(x_next, y_next)

            # Save current state
            self.prev_x = self.curr_x
            self.prev_y = self.curr_y

        # make next move
        self.curr_x = x_next
        self.curr_y = y_next
        self.path_cost = self.path_cost + 1
        self.path.append((x_next, y_next))

    def search(self, max_cost):

        end = False
        while (not end) and (self.path_cost < max_cost):

            self.debug and print(f"iteration[{self.path_cost}] x: {self.curr_x} y: {self.curr_y}")

            self.random_move()
            end, end_pos = self.check_goal()

        print(f"Final Cost: {self.path_cost}")

        return self.path_cost, end_pos

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations", 
        type=int,
        help='Number of iterations to repeat experiments')

    return parser.parse_args()


def main():
    print("Main()...")

    # Get arguements
    args = get_args()
    num_iterations = args.iterations
    
    ts = time.time()
    agent = Agent(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])

    search_limits = [1000, 10000, 100000, 1000000]
    costs = np.ndarray(shape=(len(search_limits)*num_iterations))
    n_end1 = 0
    n_end2 = 0

    # Track number of times that max limit produced a valid path
    valid_limit = {
        "1000": 0,
        "10000": 0,
        "100000": 0,
        "1000000": 0
    }

    for iter in range(num_iterations):
        print(f"[{iter}]...")
        for idx, limit in enumerate(search_limits):

            print(f"Searching: [{limit}]")
            cost, end_pos = agent.search(limit)

            if cost < limit:
                costs[idx*(iter + 1)] = cost
                valid_limit[str(limit)] = valid_limit[str(limit)] + 1
            
            if end_pos == 1:
                n_end1 = n_end1 + 1

            if end_pos == 2:
                n_end2 = n_end2 + 1

            # reset agent
            agent.reset(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])

    te = time.time()

    # Filter out non valid values
    # Filter out values near 0 and values near max floating point value
    mask = costs > 1
    costs_filtered = costs[mask]
    mask = costs_filtered < 1000000
    costs_filtered = costs_filtered[mask]
    print(costs)
    print(costs_filtered)

    # Get mean cost
    avg_cost = costs_filtered.mean()
    print(f"Average Cost: {avg_cost}, END1: {n_end1}, END2: {n_end2}")
    print(valid_limit)

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()
