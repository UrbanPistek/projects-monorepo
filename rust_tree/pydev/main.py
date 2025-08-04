import os
import time
from pprint import pprint
import matplotlib.pyplot as plt

# Internal modules
import objects as obj
from plot import plot_custom_polygons
from rtree import Rtree

# Using non-gps points for
# simple testing
SIMPLE_BOXES = [
    [[0,0], [0,12], [12,12], [12,0], [0,0]], # Contains all boxes

    [[1,3], [1,9], [5,9], [5,3], [1,3]],
    [[2,4], [2,6], [4,6], [4,4], [2,4]], # Inside the above box
    
    [[6,2], [6,6], [10,6], [10,2], [6,2]],
    [[7,3], [7,5], [8,5], [8,3], [7,3]], # Inside the above box
]

def plot_sample_data():
    fig, ax = plot_custom_polygons(SIMPLE_BOXES, colors=['blue', 'red', 'green'])
    plt.show()

def test_rtree():
    rt = Rtree(3)

    boxes = []
    for box in SIMPLE_BOXES:

        minx = min(box, key=lambda x: x[0])
        miny = min(box, key=lambda x: x[1])
        maxx = max(box, key=lambda x: x[0])
        maxy = max(box, key=lambda x: x[1])
        bl = obj.Point(minx[0], miny[1])
        tr = obj.Point(maxx[0], maxy[1])
        boxes.append(obj.Box(bl, tr))

    rt.insert(boxes[0])
    rt.insert(boxes[1])

    representation = rt.array_representation()
    print(representation)

def main():
    ts = time.perf_counter()
    
    # Plot sample data
    # plot_sample_data()

    # Test run r-tree
    test_rtree()

    te = time.perf_counter()
    print(f"Completed in {round(te-ts,3)}s")

main()
