import rclpy
from rclpy.node import Node

# ROS2 Message types
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from rclpy.qos import ReliabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry

import numpy as np

class Reader(Node):

    def __init__(self):

        super().__init__('reader')

        # imu
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imu_msg = None

        # odom
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.odom_msg = None

        # encoders
        self.encoder_sub = self.create_subscription(
            JointState, '/joint_states', self.encoder_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.encoder_msg = None
            
    def imu_callback(self, msg):
        self.imu_msg = msg
        info_list = [self.imu_msg.header.stamp.sec,
            self.imu_msg.linear_acceleration.x,
            self.imu_msg.linear_acceleration.y,
            self.imu_msg.linear_acceleration.z,
            self.imu_msg.angular_velocity.x,
            self.imu_msg.angular_velocity.y,
            self.imu_msg.angular_velocity.z,
            self.imu_msg.orientation.x,
            self.imu_msg.orientation.y,
            self.imu_msg.orientation.z,
            self.imu_msg.orientation.w]
        with open("./imu.csv", "a") as f:
            for val in range(len(info_list)):
                f.write(str(info_list[val]))
                if val < len(info_list) - 1:
                    f.write(",")
            f.write("\n")

    def odom_callback(self, msg):
        self.odom_msg = msg
        info_list = [self.odom_msg.header.stamp.sec,
            self.odom_msg.header.stamp.nanosec,
            self.odom_msg.pose.pose.position.x,
            self.odom_msg.pose.pose.position.y,
            self.odom_msg.pose.pose.position.z,
            self.odom_msg.pose.pose.orientation.x,
            self.odom_msg.pose.pose.orientation.y,
            self.odom_msg.pose.pose.orientation.z,
            self.odom_msg.pose.pose.orientation.w]
        with open("./odom.csv", "a") as f:
            for val in range(len(info_list)):
                f.write(str(info_list[val]))
                if val < len(info_list) - 1:
                    f.write(",")
            f.write("\n")

    def encoder_callback(self, msg):
        self.encoder_msg = msg

        pos = np.asarray(self.encoder_msg.position)
        vel = np.asarray(self.encoder_msg.velocity)

        info_list = [self.encoder_msg.header.stamp.sec,
            self.encoder_msg.header.stamp.nanosec,
            self.encoder_msg.name[0],
            self.encoder_msg.name[1],
            pos[0], 
            pos[1], 
            vel[0],
            vel[1],]
        with open("./encoder.csv", "a") as f:
            for val in range(len(info_list)):
                f.write(str(info_list[val]))
                if val < len(info_list) - 1:
                    f.write(",")
            f.write("\n")

def main(args=None):
    print("reader...")

    try:
        # initialize the ROS communication
        rclpy.init(args=args)

        # declare the node constructor
        reader = Reader()

        # pause the program execution, waits for a request to kill the node (ctrl+c)
        rclpy.spin(reader)

        # Explicity destroys the node
        reader.destroy_node()

        # shutdown the ROS communication
        rclpy.shutdown()

    except KeyboardInterrupt:
        print("Manual shutdown...")

if __name__ == "__main__": 
    main()
