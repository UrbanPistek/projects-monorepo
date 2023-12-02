import time 
import math
import numpy as np
import matplotlib as mpl
import random as rand

def acceptance_func(delta, t):
    return np.exp(-delta/t)

def main():

    inputs = [[60, 30], [60, 300], [40, 30], [40, 300]]

    for input in inputs:
        p = acceptance_func(input[0], input[1])
        print(f"Inputs: {input} | Acceptance: {p}")

if __name__ == "__main__":
    main()