# Assignment 3 Question 3 Solution

import time 
import math
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

X_BOUNDS = 24
Y_BOUNDS = 24

class Node:

    def __init__(self, x, y, parent=None, g=0, h=0):

        # track state
        self.x = x
        self.y = y
        self.parent = parent

        # track cost and heuristics
        self.gn = g
        self.hn = h
        self.fn = g + h

class Agent:

    def __init__(self, x0, y0, debug = False):

        # track state
        self.curr_x = x0
        self.curr_y = y0

        # track path, include start
        self.path_cost = 1
        self.path = []

        # utilities
        self.debug = debug

        # track for astar
        self.open_list = []
        self.closed_list = []
        self.fn = []

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

    def is_child_valid(self, cnode):
        
        if (cnode.x > X_BOUNDS) or (cnode.x < 0):
            return False
        
        if (cnode.y > Y_BOUNDS) or  (cnode.y < 0):
            return False

        if self.grid[cnode.y][cnode.x] == 1:
            return False

        for node in self.closed_list:
            if (cnode == node) or ((cnode.x == node.x) and (cnode.y == node.y)): 
                return False

        for node in self.open_list:
            if ((cnode == node) or ((cnode.x == node.x) and (cnode.y == node.y))) and (cnode.gn > node.gn): 
                return False

        return True

    def test_goal(self, x, y):

        if (x == LOCATIONS["end1"]["x"]) and (y == LOCATIONS["end1"]["y"]):
            print("Found: END1")
            return True

        if (x == LOCATIONS["end2"]["x"]) and (y == LOCATIONS["end2"]["y"]):
            print("Found: END2")
            return True

        return False

    def manhattan_distance(self, x0, y0, xg, yg):

        return abs(x0 - xg) + abs(y0 - yg)

    def find_min_fn_node(self):

        if len(self.open_list) == 1:
            return 0

        index = 0
        curr = self.open_list[index]

        for idx, node in enumerate(self.open_list):
            if node.fn < curr.fn:
                index = idx
                curr = node
        
        return index

    def a_star_search(self):

        # move list
        moves = [[1, 0], [0, 1], [-1, 0], [0, -1]] # right, up, left, down

        start_node = Node(self.curr_x, self.curr_y)
        self.open_list.append(start_node)

        end = False
        while not end:

            # Find node with min f(n) in open_list
            cidx = self.find_min_fn_node()
            
            curr_node = self.open_list[cidx]
            # Remove node from open list
            self.open_list.pop(cidx)

            # Add the min node to the closed_list
            self.closed_list.append(curr_node)

            # Print out state for debugging
            if self.debug:
                try: 
                    print(f"State: ({curr_node.x}, {curr_node.y}), Parent: {({curr_node.parent.x}, {curr_node.parent.y})} | Open: {len(self.open_list)} | Closed: {len(self.closed_list)}")
                except: 
                    print(f"State: ({curr_node.x}, {curr_node.y}) | Open: {len(self.open_list)} | Closed: {len(self.closed_list)}")

            # Check goal
            if self.test_goal(curr_node.x, curr_node.y):
                # Extract path 
                inode = curr_node
                while inode is not None: 
                    self.path.append([inode.x, inode.y])
                    self.path_cost = self.path_cost + 1
                    inode = inode.parent
                
                # Break out of the main loop
                break

            # Childern generation
            for idx, move in enumerate(moves): 

                # Next 
                xn = curr_node.x + move[0]
                yn = curr_node.y + move[1]

                # Calculate for both goals
                hn1 = self.manhattan_distance(xn, yn, LOCATIONS["end1"]["x"], LOCATIONS["end1"]["y"])
                hn2 = self.manhattan_distance(xn, yn, LOCATIONS["end2"]["x"], LOCATIONS["end2"]["y"])

                hn = min(hn1, hn2)
                gn = curr_node.gn + 1

                # Create new child node
                new_child = Node(xn, yn, parent=curr_node, g=gn, h=hn)

                if self.is_child_valid(new_child):
                    self.open_list.append(new_child)

        print(f"Final Cost: {self.path_cost}")
        print(f"Path: {self.path}")

    def visualize_path(self, search_type):

        print("Visualizing...")

        grid = self.grid

        for node in self.closed_list:
            grid[node.y][node.x] = 0.6

        # Determine number of nodes visited
        closed = 0
        for i in range(0, 24):
            for j in range(0, 24):
                if grid[i][j] == 0.6:
                    closed = closed + 1

        print(f"Nodes Explored: {closed}")

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
        cbar = plt.colorbar(img, cmap=cmap)
        
        # Annotate color bar
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['Valid','Special','Visited','Invalid']):
            cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation=270, color='y', fontweight='bold')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Tile Type', rotation=270)

        plt.title(f"Path: {search_type}")
        plt.legend(["Path"])

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
        help='')

    return parser.parse_args()

def main():
    print("Main()...")

    # Get arguements
    args = get_args()
    type = args.type
    
    ts = time.time()

    agent = Agent(LOCATIONS["start"]["x"], LOCATIONS["start"]["y"])
    print("A* Search...")

    agent.a_star_search()
    agent.visualize_path("A Star Search")

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()
