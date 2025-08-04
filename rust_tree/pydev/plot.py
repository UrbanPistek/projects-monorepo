import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection

# Example with custom polygons and colors
def plot_custom_polygons(polygons, colors=None, alpha=0.6):
    """
    Plot custom polygons with specified colors.
    
    Parameters:
    polygons: list of numpy arrays, each array contains vertices of a polygon
    colors: list of colors (optional)
    alpha: float, transparency value (optional)
    """
    fig, ax = plt.subplots()
    p = LineCollection(polygons, alpha=alpha) # Just show the edges
    
    if colors:
        p.set_color(colors)
    
    ax.add_collection(p)
    
    # Automatically set limits based on polygon vertices
    all_coords = np.concatenate(polygons)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.grid(True)
    ax.set_aspect('equal')
    plt.title('Custom Polygons')
    
    return fig, ax