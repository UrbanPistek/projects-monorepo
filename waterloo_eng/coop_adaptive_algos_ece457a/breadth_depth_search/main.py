# Assignment 1 Question 3 Solution

from asyncio import queues
import os 
import math
import time 
import random as rand
import numpy as np
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt

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

        '''
        Note that the first row in the 2D list is the y = 0 row 
        (i.e. bottom-most row in the maze figure). 
        '1' indicates that the node is blocked, '0' indicates that it is free.
        '''
        self.grid = \
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

        print(f"Agent Initialized: ({self.curr_x}, {self.curr_y })")

    def is_move_valid(self, x, y):
        
        if (x > X_BOUNDS) or (x < 0):
            return False
        
        if (y > Y_BOUNDS) or  (y < 0):
            return False

        # MAZE is indexed by (row, col) => (y, x)
        if self.grid[y][x] == 1:
            return False

        # MAZE is indexed by (row, col) => (y, x)
        for prev in self.path:
            if (x == prev[0]) and (y == prev[1]): 
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
        self.visited = [(x, y)]

        print(f"Reset: {self.curr_x}, {self.curr_y} || Cost: {self.path_cost} || Path: {self.path} || Visited: {self.visited}")

    def breadth_first_search(self):

        queue = []

        end = False
        while not end:

            self.debug and print(f"iteration[{self.path_cost}] x: {self.curr_x} y: {self.curr_y}")

            # Add values to queue
            queue.append([(self.curr_x-1), (self.curr_y)]) # Left
            queue.append([(self.curr_x), (self.curr_y+1)]) # Up
            queue.append([(self.curr_x+1), (self.curr_y)]) # Right
            queue.append([(self.curr_x), (self.curr_y-1)]) # Down

            # Check that each move is valid
            # Clean up stack of all historical moves already visited
            for move in queue:

                valid = self.is_move_valid(move[0], move[1])
                if not valid:
                    queue.remove(move)

            # Make next move
            next = queue.pop(0)
            self.curr_x = next[0]
            self.curr_y = next[1]

            # Save to path
            self.path_cost = self.path_cost + 1
            self.path.append((self.curr_x, self.curr_y))

            if self.path_cost > 1000: 
                break

            end, end_pos = self.check_goal()

        print(f"Final Cost: {self.path_cost}")
        print(f"Nodes Explored: {self.path_cost + len(queue)}")

        return self.path_cost, end_pos

    def depth_first_search(self):

        stack = []

        end = False
        while not end:

            self.debug and print(f"iteration[{self.path_cost}] x: {self.curr_x} y: {self.curr_y}")

            # Add values to queue
            moves = []
            moves.append([(self.curr_x), (self.curr_y-1)]) # Right
            moves.append([(self.curr_x-1), (self.curr_y)]) # Up
            moves.append([(self.curr_x), (self.curr_y+1)]) # Left
            moves.append([(self.curr_x+1), (self.curr_y)]) # Down
            
            # Check that each move is valid
            for move in moves:

                valid = self.is_move_valid(move[0], move[1])
                if valid:
                    stack.append(move)

            # Clean up stack of all historical moves already visited
            for move in stack:
                valid = self.is_move_valid(move[0], move[1])
                if not valid:
                    stack.remove(move)

            # Make next move
            next = stack.pop()
            self.curr_x = next[0]
            self.curr_y = next[1]

            # Save to path
            self.path_cost = self.path_cost + 1
            self.path.append((self.curr_x, self.curr_y))

            end, end_pos = self.check_goal()

        print(f"Final Cost: {self.path_cost}")
        print(f"Nodes Explored: {self.path_cost + len(stack)}")

        return self.path_cost, end_pos

    def visualize_path(self, search_type):

        print("Visualizing...")

        grid = self.grid

        for path in self.path:
            grid[path[1]][path[0]] = 0.6

        # configure plot
        fig = plt.figure(figsize=(16,9))
        plt.axis([0, 24, 0, 24])  
        cmap = mpl.colors.ListedColormap(['green','black','red','blue'])

        # Plot path
        path = np.array(self.path)
        xs, ys = path.T
        line = plt.plot(xs, ys,'k',label='ployline',lw=2,marker='x',mew=3)
        
        # Show start and end points
        grid[11][2] = 0.4 # start
        grid[19][23] = 0.4 # end1
        grid[21][2] = 0.4 # end2

        img = plt.imshow(grid, cmap = cmap) 
        plt.colorbar(img, cmap=cmap)  
        plt.title(f"Path: {search_type}")

        plt.text(1.5,11,'start', c='w')
        plt.text(22.5,19,'end1', c='w')
        plt.text(1.5,21,'end2', c='w')

        file_name = "".join(search_type.split())
        plt.savefig(f'{file_name}.png')
        plt.show()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type", 
        type=int,
        help='0 = Breadth First, 1 = Depth First, 2 = All')

    return parser.parse_args()


def main():
    print("Main()...")

    # Get arguements
    args = get_args()
    type = args.type
    
    ts = time.time()

    if type == 0: 
        agent = Agent(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])
        print("Breadth First Search...")
        end, end_pos = agent.breadth_first_search()
        agent.visualize_path("Breadth First Search")

    elif type == 1: 
        agent = Agent(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])
        print("Depth First Search...")
        end, end_pos = agent.depth_first_search()
        agent.visualize_path("Depth First Search")

    elif type == 2: 

        agentBF = Agent(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])
        print("Breadth First Search...")
        end, end_pos = agentBF.breadth_first_search()
        agentBF.visualize_path("Breadth First Search")

        agentDF = Agent(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])
        print("Depth First Search...")
        end, end_pos = agentDF.depth_first_search()
        agentDF.visualize_path("Depth First Search")

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()
