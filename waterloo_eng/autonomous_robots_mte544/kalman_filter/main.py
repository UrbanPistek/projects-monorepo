import pandas as pd
import numpy as np
import random as rnd
from matplotlib import pyplot as plt

from kalman import KalmanFilter
from preprocess import *

np.set_printoptions(precision=3, suppress=True)

ODOM_CSV = "./path/odom.csv"
IMU_CSV = "./path/imu.csv"
ENCODER_CSV = "./path/encoder.csv"

def plot_results(estimated_states, predicted_states, odom_df, odom_rnd, kalman_gains)-> None:

        # extract estimated and predicted x & y values
        est = np.asarray(estimated_states)
        xse, yse = est.T
        pred = np.asarray(predicted_states)
        xsp, ysp = pred.T
        kns = np.asarray(kalman_gains)
        pkn, vkn, akn = kns.T

        # Print out MSE
        est_mse_x = np.square(np.subtract(odom_df['x'], xse)).mean()
        est_mse_y = np.square(np.subtract(odom_df['y'], yse)).mean()
        print(f"MSE: x [{est_mse_x}], MSE: y [{est_mse_y}]")

        fig1, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12,7), sharex=False)

        # x and y
        ax1.plot(odom_df['x'], odom_df['y'], c='k', label='pos')
        # ax1.plot(odom_rnd['x'], odom_rnd['y'], c='m', label='mes')

        # estimates
        ax1.set_title("Kalman Filter Results")
        ax1.plot(xse, yse, c='g', label='est')
        ax1.plot(xsp, ysp, c='r', label='pred')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        # kalman gain
        ax2.plot(odom_df['sec'], pkn, c='r', label='pos Kn')
        ax2.plot(odom_df['sec'], vkn, c='g', label='vel Kn')
        ax2.plot(odom_df['sec'], akn, c='b', label='acc Kn')
        ax2.set_xlabel("ts")
        ax2.set_ylabel("Kn")
        ax2.legend()

        plt.show()

def main():
    print("Kalman Filter...")

    # import and pre-process data
    imu_df, odom_df, encoder_df, odom_rnd_df = load_data("./path/imu.csv", "./path/odom.csv", "./path/encoder.csv")
    # plot_data_imu(imu_df)
    # plot_data_odom(odom_df)
    # plot_data_encoder(encoder_df)
    # plot_ground_truth_vs_noisy(odom_df, odom_rnd_df)

    # =================================
    # kalman
    # =================================

    # Run the filter
    kalman = KalmanFilter(odom_rnd_df, imu_df)
    kalman.run()

    # Plot results
    ests = kalman.estimated_states
    preds = kalman.predicted_states
    kns = kalman.kalman_gains
    plot_results(ests, preds, odom_df, odom_rnd_df, kns)

if __name__ == "__main__":
    main()