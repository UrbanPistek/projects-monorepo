import copy
import pandas as pd
import numpy as np
import random as rnd
from matplotlib import pyplot as plt
from tabulate import tabulate

from kalman import KalmanFilter

np.set_printoptions(precision=3, suppress=True)

def load_data(imu_csv: str, odom_csv: str, encoder_csv: str) -> pd.DataFrame:


    # =================================
    # Read Data
    # =================================

    odom = pd.read_csv(odom_csv, names=['sec', 'nano_sec', 'x', 'y', 'z','qx', 'qy', 'qz', 'qw'])
    imu = pd.read_csv(imu_csv, names=['sec', 'ax', 'ay', 'az', 'omega_x', 'omega_y', 'omega_z', 'qx', 'qy', 'qz', 'qw'])
    encoder = pd.read_csv(encoder_csv, names=['sec', 'nano_sec', 'name1', 'name2', 'theta_w_left', 'theta_w_right', 'ul', 'ur'])

    print(f"\n{tabulate(odom.head(), headers='keys', tablefmt='fancy_grid')}")
    print(f"\n{tabulate(imu.head(), headers='keys', tablefmt='fancy_grid')}")
    print(f"\n{tabulate(encoder.head(), headers='keys', tablefmt='fancy_grid')}")

    # drop nan
    odom.dropna(inplace=True)
    imu.dropna(inplace=True)
    encoder.dropna(inplace=True)

    # =================================
    # Preprocess
    # =================================

    # Group by ts and get mean of values
    imu['ts'] = imu['sec']
    odom['ts'] = odom['sec']
    encoder['ts'] = encoder['sec']
    imu_df = imu.groupby(by=['ts'], group_keys=False, as_index=True).mean()
    odom_df = odom.groupby(by=['ts'], group_keys=False, as_index=True).mean()
    encoder_df = encoder.groupby(by=['ts'], group_keys=False, as_index=True).mean()

    # Get variance of values
    odom_df['x_var'] = odom.groupby(by=['ts'], group_keys=False)['x'].var()
    odom_df['y_var'] = odom.groupby(by=['ts'], group_keys=False)['y'].var()
    imu_df['ax_var'] = imu.groupby(by=['ts'], group_keys=False)['ax'].var()
    imu_df['ay_var'] = imu.groupby(by=['ts'], group_keys=False)['ay'].var()
    encoder_df['ur_var'] = encoder.groupby(by=['ts'], group_keys=False)['ur'].var()
    encoder_df['ul_var'] = encoder.groupby(by=['ts'], group_keys=False)['ul'].var()

    # shift sec in imu to start at 0
    imu_df['sec'] = imu_df['sec'] - 9.0
    odom_df['sec'] = odom_df['sec'] - 9.0
    encoder_df['sec'] = encoder_df['sec'] - 9.0

    # Print mean acceleration variance
    print(f"ax mean var: {imu_df['ax_var'].mean()}")
    print(f"ay mean var: {imu_df['ay_var'].mean()}")
    print(f"wz mean var: {imu_df['omega_z'].mean()}")

    # Ensure both sets are the same length
    odom_df = odom_df[:][:24]
    encoder_df = encoder_df[:][:24]

    # =================================
    # Create Noisy Odom Set
    # =================================

    odom_noisy_df = copy.deepcopy(odom_df)

    # add gaussian noise to data
    mu, sigma = 0, 0.01
    # creating a noise with the same dimension as the dataset (2,2) 
    noise = np.random.normal(mu, sigma, [2, len(odom_noisy_df)])

    # set variance
    odom_noisy_df['x_var'] = sigma**2
    odom_noisy_df['y_var'] = sigma**2

    odom_noisy_df['x_noise'] = noise[0]
    odom_noisy_df['y_noise'] = noise[1]

    odom_noisy_df['x'] = odom_noisy_df['x'] + odom_noisy_df['x_noise']
    odom_noisy_df['y'] = odom_noisy_df['y'] + odom_noisy_df['y_noise']

    # All dataframes should have the sample length
    print(f"\n==>odom:")
    print(f"{odom_df.info()}")
    print(f"{odom_df.head()}")

    print(f"\n==>imu:")
    print(f"{imu_df.info()}")
    print(f"{imu_df.head()}")

    print(f"\n==>encoder:")
    print(f"{encoder_df.info()}")
    print(f"{encoder_df.head()}")

    print(f"\n==>noisy odom:")
    print(f"{odom_noisy_df.info()}")
    print(f"{odom_noisy_df.head()}")

    return imu_df, odom_df, encoder_df, odom_noisy_df

def plot_data_imu(df: pd.DataFrame) -> None: 
    
    # configure plot
    fig1, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(12,7), sharex=True)

    # x over time
    ax1.set_title(f"acceleration")
    ax1.plot(df['sec'], df['ax'], 'c', label='ax')
    ax1.plot(df['sec'], df['ay'], 'm', label='ay')
    ax1.set_ylabel("acceleration")
    ax1.legend()

    ax2.set_title(f"omega")
    ax2.plot(df['sec'], df['omega_z'], 'c', label='oz')
    ax2.set_ylabel("velocity")
    ax2.set_xlabel("ts")
    ax2.legend()

    plt.show()

def plot_data_encoder(df: pd.DataFrame) -> None: 
    
    # configure plot
    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=False)

    # x over time
    ax1.set_title(f"wheel velocities")
    ax1.plot(df['sec'], df['ur'], 'c', label='ur')
    ax1.plot(df['sec'], df['ul'], 'm', label='ul')
    ax1.set_ylabel("velocity")
    ax1.set_xlabel("ts")
    ax1.legend()

    plt.show()

def plot_data_odom(df: pd.DataFrame) -> None: 
    
    # configure plot
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 1]}, figsize=(12,7), sharex=False)

    # x over time
    ax1.set_title(f"position")
    ax1.scatter(df['sec'], df['x'], c='c', label='x', marker='o')
    ax1.set_ylabel("x")
    ax1.set_xlabel("s")
    ax1.legend()

    # y over time
    ax2.scatter(df['sec'], df['y'], c='m', label='y', marker='o')
    ax2.set_ylabel("y")
    ax2.set_xlabel("s")
    ax2.legend()

    # x and y
    ax3.scatter(df['x'], df['y'], c='r', label='pos', marker='x')
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()

    plt.show()

def plot_ground_truth_vs_noisy(odom: pd.DataFrame, odom_rnd: pd.DataFrame) -> None:

    fig1, (ax1) = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, figsize=(12,7), sharex=False)

    # x and y
    ax1.scatter(odom['x'], odom['y'], c='k', label='truth', marker='x')
    ax1.scatter(odom_rnd['x'], odom_rnd['y'], c='r', label='input', marker='o')

    ax1.set_title("Ground Truth vs Noisy Odom")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    plt.show()