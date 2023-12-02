import pandas as pd

from math import atan2, sqrt, sin, cos
from random import gauss

# Custom modules
from state import State
from utils import euler_from_quaternion

def load_odometry_data(csv_path="./bagreader/odom_point1.csv") -> list[tuple[State, State]]:

    # Threshold on number of items to read from csv
    thres = 330
    if "point5" in csv_path:
        thres = 360

    # Input data format [s, ns, x, y, z, x, y, z, w]
    df = pd.read_csv(csv_path, index_col=False, header=None)
    df = df.fillna(0) # replace any possible nan's with a 0
    data = df.to_numpy()

    odom_states = []
    # loop through rest of data and create state objects
    # downsample by incrememting by 3
    for idx in range(0, thres, 3):

        # Generate current state
        xc = data[idx][2]
        yc = data[idx][3]
        _, _, thetac = euler_from_quaternion(data[idx][5], data[idx][6], data[idx][7], data[idx][8])
        curr_state = State(xc, yc, thetac)

        # Only for the 1st value is the prev state equal to the current state
        if idx == 0:
            prev_state = curr_state
        else:
            # get previous state from last entry
            xc = data[idx-1][2]
            yc = data[idx-1][3]
            _, _, thetac = euler_from_quaternion(data[idx-1][5], data[idx-1][6], data[idx-1][7], data[idx-1][8])
            prev_state = State(xc, yc, thetac)

        odom_states.append((curr_state, prev_state))

    return odom_states

class SampleOdomModel:

    def __init__(self, alpha1=0.1, alpha2=0.1, alpha3=0.1, alpha4=0.1) -> None:

        # Initialize alpha values for adjusting noise 
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        # For gaussian
        self.mu = 0 # mean
        self.sigma = 1 # standard deviation

    def calc_odom_sample(self, xt_prev: State, ut: tuple[State, State]) -> State:

        # State objects
        xb_curr = ut[1] # previous relative odom
        xb_prev = ut[0] # current relative odom
        
        # Calculate first rotation
        rot1 = atan2((xb_curr.y - xb_prev.y), (xb_curr.x - xb_prev.x)) - xb_prev.theta

        # Calculate translation
        trans = sqrt((xb_prev.x - xb_curr.x)**2 + (xb_prev.y - xb_curr.y)**2)

        # Calculate second rotation
        rot2 = xb_curr.theta - xb_prev.theta - rot1

        # Add noise 
        rot1_hat = rot1 + gauss(self.mu, self.sigma)*(self.alpha1*rot1 + self.alpha2*trans)
        trans_hat = trans + gauss(self.mu, self.sigma)*(self.alpha3*trans + self.alpha4*(rot1 + rot2))
        rot2_hat = rot2 + gauss(self.mu, self.sigma)*(self.alpha1*rot2 + self.alpha2*trans)

        # Predicted state
        x = xt_prev.x + trans_hat*cos(xt_prev.theta + rot1_hat)
        y = xt_prev.y + trans_hat*sin(xt_prev.theta + rot1_hat)
        theta = xt_prev.theta + rot1_hat + rot2_hat
        xt = State(x, y, theta)

        return xt

# Test function to test the sample odom model 
def test_sample_odom() -> None:
    print("\ntest_sample_odom()")

    xt_prev = State(0.00000004254, 0.00000000003253, 1.3254235352)
    ut = (State(0.00000000320200, 0.00000000312326, 1.3298742283), State(0.00000000320200, 0.00000000312326, 1.32987533790221))

    odomModel = SampleOdomModel()
    xt = odomModel.calc_odom_sample(xt_prev, ut)

    print(f"Predicted State: \nx:{xt.x}, y:{xt.y}, theta:{xt.theta}")

def test_load_odometry_data() -> None:
    print("\ntest_load_odometry_data()")
    odom_data = load_odometry_data(csv_path="./bagreader/odom_point1.csv")
    print(f"point1 data: len: {len(odom_data)} row: {odom_data[0]}")

    odom_data = load_odometry_data(csv_path="./bagreader/odom_point5.csv")
    print(f"point5 data: len: {len(odom_data)} row: {odom_data[0]}")
    print(f"point5-state: (x, y, theta): ({odom_data[0][0].x}, {odom_data[0][0].y}, {odom_data[0][0].theta})")