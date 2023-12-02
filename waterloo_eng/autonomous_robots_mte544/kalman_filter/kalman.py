import pandas as pd
import numpy as np
from numpy.linalg import inv

np.set_printoptions(precision=3, suppress=True)

class KalmanFilter():

    def __init__(self, odom: pd.DataFrame, imu: pd.DataFrame) -> None:

        # =================================
        # State
        # =================================
        
        # time step - based on input data time steps 
        self.dt = 0.034

        # state vector
        # [xn vel_xn acc_xn yn vel_yn acc_yn]
        self.xhat = np.matrix([0, 0, 0, 0, 0, 0]).transpose()

        # State transition matrix
        self.F = np.matrix([
            [1, self.dt, (self.dt**2)/2, 0, 0,       0             ],
            [0, 1,       self.dt,        0, 0,       0             ],
            [0, 0,       1,              0, 0,       0             ], 
            [0, 0,       0,              1, self.dt, (self.dt**2)/2],
            [0, 0,       0,              0, 1,       self.dt       ], 
            [0, 0,       0,              0, 0,       1             ],
            ])

        # Covariance matrix
        # Initialize using identity matrix
        # Assume values in x & y are not correlated => covariance = 0
        # Assume (x, vx) and (y, vy) are correlated => covariance = non-zero
        # Initialize with a very high value
        self.P = 10*np.identity(6)

        # Acceleration variance, based on input data
        self.acc_sigma = 0.075

        # Process noise matrix
        self.Q = self.acc_sigma*np.identity(6)

        # Measurement vector
        self.zn = np.matrix([0, 0]).transpose()

        # Obseration matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        # Measurement uncertainty matrix
        # Assume constant 
        # Assume x and y are not correlated
        # Use mean of variance across the measurements
        self.odom_x_var = odom['x_var'].mean()
        self.odom_y_var = odom['y_var'].mean()
        self.R = np.matrix([[self.odom_x_var, 0], [0, self.odom_y_var]])

        # Control input vector
        # Assume acceleration is zero to start
        self.u = np.matrix([0, 0]).transpose()

        # Control matrix
        self.G = np.matrix([
                        [0, 0],
                        [0, 0],
                        [1, 0], 
                        [0, 0], 
                        [0, 0],
                        [0, 1],
                        ])


        # =================================
        # Initialization
        # =================================

        # Calculate initial first prediction
        self.xhat1 = self.F*self.xhat + self.G*self.u

        self.P1 = self.F*self.P*(self.F.T) + self.Q

        # Save datasets 
        self.odom_df = odom
        self.imu_df = imu

        # Save metrics
        self.estimated_states = []
        self.predicted_states = []
        self.kalman_gains = []

    def run(self):
        
        # get timestamps
        ts = self.odom_df['sec'].values

        for k in range(0, len(ts)):

            # =================================
            # Measurements and Control
            # =================================
            
            odom = self.odom_df.iloc[k]
            imu = self.imu_df.iloc[k]

            # Get measurement values
            self.zn = np.matrix([odom['x'], odom['y']]).transpose()

            # Get control input
            self.u = np.matrix([imu['ax'], imu['ay']]).transpose()

            # =================================
            # Update
            # =================================

            # Calculate kalman gain
            self.Kn = self.P1*(self.H.T)*inv(self.H*self.P1*(self.H.T) + self.R)

            # Estimate current state
            self.xhat = self.xhat1 + self.Kn*(self.zn - self.H*self.xhat1)

            # Update estimate uncertainty
            # Using full uncertantity update equation to ensure numerical stability
            self.P = (np.identity(6) - self.Kn*self.H)*self.P1*(np.transpose(np.identity(6) - self.Kn*self.H)) + self.Kn*self.R*self.Kn.T

            # =================================
            # Predict
            # =================================
            
            # Predict next state
            self.xhat1 = self.F*self.xhat + self.G*self.u

            # Predict next covariance matrix
            self.P1 = self.F*self.P*(self.F.T) + self.Q

            # =================================
            # Tracking Metrics
            # =================================

            print(f"k:{k}")
            print(f"Kn:\n{self.Kn}")
            # print(f"state:\n{self.xhat} \ncov:\n{self.P} \npredicted state:\n{self.xhat1}")

            # Save estimatations and predictions
            xhat_n = np.asarray(self.xhat)
            xhat_n1 = np.asarray(self.xhat1)
            kns = np.asarray(self.Kn)

            # Track specific values
            self.estimated_states.append([xhat_n[0][0], xhat_n[3][0]])
            self.predicted_states.append([xhat_n1[0][0], xhat_n1[3][0]])
            self.kalman_gains.append([ np.mean([kns[0][0], kns[3][1]]), np.mean([kns[1][0], kns[4][1]]), np.mean([kns[2][0], kns[5][1]])])