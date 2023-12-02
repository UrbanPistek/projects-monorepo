# A* Algorithm

import time 
import math
import copy
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
from math import sqrt
from matplotlib import pyplot as plt

# from turtlebot4_lab3.preprocess import load_downsampled_map # running in ROS2
from preprocess import load_downsampled_map

class Node:

    def __init__(self, x: float, y: float, parent=None, g=0, h=0) -> None:

        # track state
        self.x = x
        self.y = y
        self.parent = parent

        # track cost and heuristics
        self.gn = g
        self.hn = h
        self.fn = g + h

class UnitConverter():

    def __init__(self, gshape: tuple, mo: tuple, resolution: float) -> None:
        """
        Class to converter coordinates in meters to pixel values and back 
        based on the map frame of reference.
        """

        # shape of the map grid
        self.grid_shape = gshape

        # Actual value of the origi in meters
        self.m_origin = mo
        self.resolution = resolution # resolution of the map

        # x-axis length 
        self.x_len = self.resolution * self.grid_shape[0]

        # y-axis length
        self.y_len = self.resolution * self.grid_shape[1]

    def __round_nearest(self, x: float, a: float) -> float:
        """
        x: value to round
        a: round increment, for example: 0.05

        Avoids floating point operation errors.
        """
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def m_to_pixel(self, x: float, y: float) -> tuple:
        """
        Convert coordinates in meters to pixels
        Based on the map frame of reference.
        """

        # round input values 
        x = self.__round_nearest(x, 0.05)
        y = self.__round_nearest(y, 0.05)
        
        # formula: px = ((x - xo)/xlen)*pxl
        # x = x point in meters
        # xo = x value of global origin in meters
        # xlen = length of x axis in meters
        # pxl = length of x axis in pixels
        xp = ((x - self.m_origin[0])/self.x_len)*(self.grid_shape[0])

        # formula: py = ((y - yo)/ylen)*pyl
        # y = y point in meters
        # yo = y value of global origin in meters
        # ylen = length of y axis in meters
        # pyl = length of y axis in pixels
        yp = ((y - self.m_origin[1])/self.y_len)*(self.grid_shape[1])

        # round values to nearest int
        # converting to float because grid is using floats
        xp = round(xp)
        yp = round(yp)

        return (xp, yp)

    def pixel_to_m(self, x: float, y: float) -> tuple:
        """
        Convert coordinates in meters to pixels
        Based on the map frame of reference.
        """
        
        # formula: xm = px*res + xo
        # px = x point in pixels
        # res = resolution
        # xo = x value of global origin in meters
        xm = x*self.resolution + self.m_origin[0]

        # formula: ym = py*res + yo
        # py = y point in pixels
        # res = resolution
        # yo = y value of global origin in meters
        ym = y*self.resolution + self.m_origin[1]

        return xm, ym

class AStar:

    def __init__(self, start: tuple, goal: tuple, occupation_map: np.ndarray, max_iterations=10000, debug=False, traversal_type=4) -> None:

        # track state
        self.curr_x = start[0]
        self.curr_y = start[1]

        # start
        self.x_start = start[0]
        self.y_start = start[1]

        # goal 
        self.x_end = goal[0]
        self.y_end = goal[1]

        # track path, include start
        self.path_cost = 1
        self.path = []

        # utilities
        self.debug = debug
        self.iterations = 0 # track number of iterations
        self.max_iter = max_iterations # have a hard limit

        # track for astar
        self.open_list = []
        self.closed_list = []
        self.fn = []

        # Traversal type ethier 4 or 8 point
        self.traversal_type = traversal_type

        # Setup bounds based on input map
        self.grid = occupation_map
        self.x_bounds = (self.grid.shape[0] - 1)
        self.y_bounds = (self.grid.shape[1] - 1)

    def __is_child_valid(self, cnode: Node) -> bool:
        """
        Validate that the child node. 
        """
        
        # Check if within x bounds in pixel coordinate system
        if (cnode.x > self.x_bounds) or (cnode.x < 0):
            return False
        
        # Check if within y bounds in pixel coordinate system
        if (cnode.y > self.y_bounds) or  (cnode.y < 0):
            return False

        # Check if the pixel is considered occupied
        # A value of 1 corresponds to occupied
        # Using a threshold here to avoid floating point errors
        if self.grid[cnode.x][cnode.y] > 0.9:
            return False

        # Check if the node exists in the closed list
        for node in self.closed_list:

            # using tolerance to avoid floating point rounding errors
            tol = 0.1
            dx = abs(cnode.x - node.x)
            dy = abs(cnode.y - node.y)
            if ((dx < tol) and (dy < tol)): 
                return False

        # Check if the node exists in the open list
        for node in self.open_list:

            # using tolerance to avoid floating point rounding errors
            tol = 0.1
            dx = abs(cnode.x - node.x)
            dy = abs(cnode.y - node.y)
            if ((dx < tol) and (dy < tol)) and (cnode.gn >= node.gn): 
                return False

        return True

    def __test_goal(self, x: float, y: float) -> bool:
        return ((x == self.x_end) and (y == self.y_end))

    def __manhattan_distance(self, x0: float, y0: float, xg: float, yg: float) -> float:
        return abs(x0 - xg) + abs(y0 - yg)

    def __euclidean_distance(self, x0: float, y0: float, xg: float, yg: float) -> float:
        return sqrt((x0 - xg)**2 + (y0 - yg)**2)

    def __find_min_fn_node(self) -> int:
        """
        Find node in the open list with the lowest fn value. 

        Returns the index on the node in the open list.
        """

        # Base case
        if len(self.open_list) == 1:
            return 0

        index = 0
        curr = self.open_list[index]

        for idx, node in enumerate(self.open_list):
            if node.fn < curr.fn:
                index = idx
                curr = node
        
        return index

    def search(self):
        print("searching...")

        # move list
        if self.traversal_type == 8:
            # Using 8 point traversal
            moves = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        else:
            # Using 4 point traversal
            moves = [[1, 0], [0, 1], [-1, 0], [0, -1]] # right, up, left, down

        # Initialize start node
        start_node = Node(self.curr_x, self.curr_y)
        self.open_list.append(start_node)

        pbar = tqdm(total=self.max_iter) # progress bar
        while True: 

            # Find node with min f(n) in open_list
            cidx = self.__find_min_fn_node()
            curr_node = self.open_list[cidx]
            
            # Remove node from open list
            self.open_list.pop(cidx)

            # Add the node to the closed_list
            self.closed_list.append(curr_node)
            
            # Check if at goal, if so extract full path and break out of main loop
            if self.__test_goal(curr_node.x, curr_node.y) or (self.iterations > self.max_iter):
                # Extract path 
                inode = curr_node
                while inode is not None: 
                    self.path.append([inode.x, inode.y])
                    self.path_cost = self.path_cost + 1
                    inode = inode.parent
                
                # Break out of the main loop
                break

            # Childern generation
            for _, move in enumerate(moves): 

                # Next node
                xn = curr_node.x + move[0]
                yn = curr_node.y + move[1]

                # Calculate cost
                if self.traversal_type == 8:
                    # Using 8 point traversal
                    hn = self.__euclidean_distance(xn, yn, self.x_end, self.y_end)
                else:
                    # Using 4 point traversal
                    hn = self.__manhattan_distance(xn, yn, self.x_end, self.y_end)

                gn = curr_node.gn + 1

                # Create new child node
                new_child = Node(xn, yn, parent=curr_node, g=gn, h=hn)

                # If the child node is valid add to the open list
                if self.__is_child_valid(new_child):
                    self.open_list.append(new_child)
            
            # track number of iterations
            self.iterations += 1
            pbar.update(1)

        del pbar

    def visualize_path(self, show=False) -> None:
        """
        Visualize A* solution on grid.
        """

        print("Visualizing...")

        grid = copy.deepcopy(self.grid)

        # added checked nodes
        for node in self.closed_list:
            grid[node.x][node.y] = 0.5

        # configure plot
        plt.figure(figsize=(12,7))
       
        # define custom color map
        cmap = mpl.colors.ListedColormap(['cyan','magenta','blue'])

        # Plot path
        path = np.array(self.path)
        xs, ys = path.T
        plt.plot(xs, ys,'k', label='path', lw=2, mew=3)
        
        # Show start and end points
        plt.scatter(self.x_start, self.y_start, marker='x', color='k', label="start")
        plt.scatter(self.x_end, self.y_end, marker='p', color='k', label="end")

        # transpose the grid to display properly with imshow()
        display_grid = copy.deepcopy(grid.T)
        img = plt.imshow(display_grid, cmap = cmap, origin='lower') 
        cbar = plt.colorbar(img, cmap=cmap)
        
        # Annotate color bar
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['Valid','Visited', 'Invalid']):
            cbar.ax.text(.5, (2 * j + 1) / 6, lab, ha='center', va='center', rotation=270, color='k', fontweight='bold')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Tile Type', rotation=270)

        plt.title(f"A* Solution Down-sampled Map")
        plt.xlabel(f"x")
        plt.ylabel(f"y")
        plt.legend()

        if show:
            file_name = "astar_solution"
            plt.savefig(f'{file_name}.png')
            plt.show()

    def convert_path_to_meters(self, uc: UnitConverter) -> np.ndarray:
        """
        Convert full pixel path to meters based on the unit converter.
        """
        
        meter_path = []
        for point in self.path:
            xm, ym = uc.pixel_to_m(point[0], point[1])

            # Add offset to all values
            xm = xm + uc.resolution/2
            ym = ym + uc.resolution/2

            meter_path.append([xm, ym])

        return np.asarray(meter_path,dtype=float)

def run_astar(map_file: str, map_params_file: str, start: tuple, end: tuple) -> np.ndarray:
    print("Running A*...")

    # Preprocess map data
    # Applies downsampling based on the reduction factor and applies a binary threshold
    reduction_factor = 8
    binary_grid, params = load_downsampled_map(map_file, map_params_file, reduction_factor=reduction_factor)
    
    # Configure unit converter based on map shape and reduction factor
    grid_shape = binary_grid.shape
    max_iter = grid_shape[0]*grid_shape[1]
    uc = UnitConverter(grid_shape, params['origin'][:2], params['resolution']*reduction_factor)

    # Covert start and end coordinates to pixel values based on the unit converter
    map_start_px = uc.m_to_pixel(start[0], start[1])
    map_end_px = uc.m_to_pixel(end[0], end[1])

    # Run A*
    astar = AStar(map_start_px, map_end_px, binary_grid, max_iterations=max_iter, debug=False)
    astar.search()

    # Convert path in pixel values to global reference frame in meters
    path = astar.convert_path_to_meters(uc)

    # invert so that start point is at index[0]
    path = np.flip(path, axis=0)

    return path

def main():
    print("A* Search...")
    
    ts = time.time()

    file = "./maps/map.pgm"
    params_file = "./maps/map.yaml"
    map_start_m = (3.0, -2.47)
    map_end_m = (0.0, -0.5)
    path = run_astar(file, params_file, map_start_m, map_end_m)
    print(f"Final Path:\n{path}")

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()
