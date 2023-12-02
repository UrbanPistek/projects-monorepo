import time 
import numpy as np
from matplotlib import pyplot as plt

from preprocess import visualize_map, pool2d, load_downsampled_map
from search import AStar, UnitConverter

def manhattan_distance(x0: float, y0: float, xg: float, yg: float) -> float:
    return abs(x0 - xg) + abs(y0 - yg)

def test_map_downsampling() -> None:
    
    file = "./maps/map.pgm"
    map = plt.imread(file)

    map_alt = np.where(map > 100, 255, 0)
    map_down = pool2d(map_alt, 4, 4, pool_mode='min')
    map_bin = np.where(map_down > 100, 0, 1)
    map_bin = map_bin.astype(float)

    visualize_map(map.T, "Original")
    visualize_map(map_alt.T, "Mapped")
    visualize_map(map_down.T, "Downsampled")
    visualize_map(map_bin.T, "Downsampled Binary")

def test_threshold() -> None:
    
    file = "./maps/Entry_1.pgm"
    map = plt.imread(file)

    map_alt = np.where(map > 100, 0, 1)

    visualize_map(map.T, "Original")
    visualize_map(map_alt.T, "Mapped")

def test_unit_converter():
    print("test_unit_converter()...")

    file = "./maps/map.pgm"
    params_file = "./maps/map.yaml"
    binary_grid, params = load_downsampled_map(file, params_file, show=False, reduction_factor=1)
    grid_shape = binary_grid.shape

    uc = UnitConverter(grid_shape, params['origin'][:2], params['resolution'])

    xp, yp = uc.m_to_pixel(1, 1)
    xm, ym = uc.pixel_to_m(135, 94)

    print(f"(1, 1) meter to pixel = ({xp}, {yp}) | Assert: {xp == 135 and yp == 94}")
    print(f"(135, 94) meter to pixel = ({xm}, {ym}) | Assert: {xm == 1 and ym == 0.98}")

def test_astar_meter_input_downsampled() -> None:
    print("test_astar_meter_input()...")

    file = "./maps/Entry_1.pgm"
    params_file = "./maps/Entry_1.yaml"
    reduction_factor = 8
    binary_grid, params = load_downsampled_map(file, params_file, reduction_factor=reduction_factor)

    grid_shape = binary_grid.shape
    max_iter = grid_shape[0]*grid_shape[1]

    uc = UnitConverter(grid_shape, params['origin'][:2], params['resolution']*reduction_factor)

    # map_start_m = (-4.5, -3.25)
    # map_start_m = (2.5, 2)
    # map_start_m = (-4.5, 3)
    # map_start_m = (7, 4)

    # map_end_m = (0.0, -0.5)
    # map_end_m = (7, -3.5)
    # map_end_m = (1, -3.5)

    # Entry 1
    map_start_m = (0, 0)
    map_end_m = (-2.71, 0)

    map_start_px = uc.m_to_pixel(map_start_m[0], map_start_m[1])
    map_end_px = uc.m_to_pixel(map_end_m[0], map_end_m[1])

    # astar = AStar(map_start_px, map_end_px, binary_grid, max_iterations=max_iter, debug=False, traversal_type=8)
    astar = AStar(map_start_px, map_end_px, binary_grid, max_iterations=max_iter, debug=False)
    
    try:
        astar.search()
        astar.visualize_path(show=True)
        path = astar.convert_path_to_meters(uc)

        print(f"map_start_m: {map_start_m}")
        print(f"map_end_m: {map_end_m}")
        print(f"cost: {astar.path_cost*params['resolution']*reduction_factor}")
        print(f"manhattan dist: {manhattan_distance(map_start_m[0], map_start_m[1], map_end_m[0], map_end_m[1])}")

        # convert path back to pixels on original size to visualize
        map = plt.imread(file)
        map = np.flip(map, axis=0) # need to flip axis

        _, params = load_downsampled_map(file, params_file, reduction_factor=1)
        grid_shape = binary_grid.shape
        max_iter = grid_shape[0]*grid_shape[1]
        uc = UnitConverter(grid_shape, params['origin'][:2], params['resolution'])

        pixel_path = []
        for point in path:
            xm, ym = uc.m_to_pixel(point[0], point[1])

            # Add offset to all values
            xm = xm + uc.resolution/2
            ym = ym + uc.resolution/2

            pixel_path.append([xm, ym])

        pixel_path = np.asarray(pixel_path,dtype=int)

        # configure plot
        fig, ax = plt.subplots(figsize=(12,7))

        # Plot path
        xs, ys = pixel_path.T
        plt.plot(xs, ys,'k', label='path', lw=2, mew=3)
        
        # Show start and end points
        plt.scatter(pixel_path[-1][0], pixel_path[-1][1], marker='x', color='k', label="start")
        plt.scatter(pixel_path[0][0], pixel_path[0][1], marker='p', color='k', label="end")

        img = plt.imshow(map, cmap = "inferno", origin='lower') 
        plt.colorbar(img, cmap="inferno")
        plt.title(f"A* Solution")
        plt.xlabel(f"x")
        plt.ylabel(f"y")
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        print("\ncancelled...")

def main():
    print("A* Tests...")
    
    ts = time.time()

    # test_threshold()
    # test_failed_astar_input()
    # test_astar_pixel_input()
    # test_unit_converter()
    # test_map_downsampling()
    # test_astar_meter_input()
    test_astar_meter_input_downsampled()

    te = time.time()

    print(f"Elapsed time: {te - ts}s")

if __name__ == "__main__":
    main()