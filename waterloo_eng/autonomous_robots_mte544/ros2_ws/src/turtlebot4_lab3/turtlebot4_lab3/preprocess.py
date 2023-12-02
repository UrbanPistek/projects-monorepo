import yaml
import numpy as np
from yaml.loader import SafeLoader
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided

def visualize_map(map: np.ndarray, title: str) -> None:
    """
    Simply plot and visualize map data.
    """

    img = plt.imshow(map, cmap = "inferno", origin='lower') 
    plt.colorbar(img, cmap="inferno")
    plt.title(title)
    plt.xlabel(f"x")
    plt.ylabel(f"y")
    plt.show()

def map_histogram(map: np.ndarray) -> None:
    """
    Generate a histogram of the map data.
    """

    plt.hist(map, bins=51) 
    plt.title("Map Histogram")
    plt.xlabel(f"bin")
    plt.ylabel(f"count")
    plt.show()

def pool2d(map: np.ndarray, kernel_size, stride, padding=0, pool_mode='min')-> np.ndarray:
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'min', 'max' or 'avg'
    '''
    # Padding
    A = np.pad(map, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'min':
        return A_w.min(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

def downsample_map(map: np.ndarray, reduction_size=4, free_threshold=0.25) -> np.ndarray:
    """
    Perform a down sampling operation using min pooling. 

    A kernel of size (reduction_size, reduction_size) is used to perfom a 
    min pooling operation on the map, effecetively reducing the dimensionality by 4.

    Returns the map as a nd.array with the binary threshold applied.
    """

    # Map all values which are likely to be free to ~255
    map_alt = np.where(map > round(free_threshold*255), 255, 0)

    # Perform the min pooling
    map_down = pool2d(map_alt, reduction_size, reduction_size, pool_mode='min')

    # Apply a binary threshold where all values above 100 are 0 (Free)
    # All values below free_threshold are mapped to 1 (Occupied)
    # This binary threshold mapping is based on the map data provided
    map_bin = np.where(map_down > round(free_threshold*255), 0, 1)

    # Ensure the data is a floating point type 
    map_bin = map_bin.astype(float)

    return map_bin

def load_downsampled_map(map_file: str, map_params_file: str, reduction_factor=4) -> np.ndarray:
    """
    Load map data - including the map istelf and the parameters. 

    Then perform a downsampling operation based on the reduction factor.

    Using a reduction factor of 1 returns the map with no downsampling. 

    Returns the map in a binary occupancy format.
    """

    # load map params
    map_params = None
    with open(map_params_file) as f:
        map_params = yaml.load(f, Loader=SafeLoader)

    # load map and perform downsampling
    map = plt.imread(map_file)
    map_bin = downsample_map(map, reduction_factor, free_threshold=map_params["free_thresh"])

    # transpose data to align with global coordinate system
    map_bin = map_bin.T
    map_bin = np.flip(map_bin, axis=1)

    return map_bin, map_params