import rclpy
# import the ROS2 python libraries
from rclpy.node import Node
# import the Twist interface from the geometry_msgs package
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from rclpy.qos import ReliabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry


class Bagreader(Node):

    def __init__(self):
        # Here you have the class constructor
        # call super() in the constructor to initialize the Node object
        # the parameter you pass is the node name
        super().__init__('bagreader')
        # create the subscriber object
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        # define the timer period for 0.5 seconds
        # self.timer_period = 0.5
        # define the variable to hold the received laser data
        self.laser_forward = 0


    # need to setup the laser callback message

    def laser_callback(self, msg):
        self.laser_forward = msg
        print(self.laser_forward)
        info_list = [self.laser_forward.header.stamp.sec,
                     self.laser_forward.header.stamp.nanosec,
                     self.laser_forward.angle_min,
                     self.laser_forward.angle_max,
                     self.laser_forward.angle_increment,
                     self.laser_forward.time_increment,
                     self.laser_forward.scan_time,
                     self.laser_forward.range_min,
                     self.laser_forward.range_max,
                     *self.laser_forward.ranges]
        with open("./src/turtlebot4_lab2/bagreader/scan.csv", "a") as f:
            for val in range(len(info_list)):
                f.write(str(info_list[val]))
                if val < len(info_list) - 1:
                    f.write(",")
            f.write("\n")

        # this takes the value at angle 359 (equivalent to angle 0)

    def odom_callback(self, msg):
        self.odom = msg
        
        info_list = [self.odom.header.stamp.sec,
                     self.odom.header.stamp.nanosec,
                     self.odom.pose.pose.position.x,
                     self.odom.pose.pose.position.y,
                     self.odom.pose.pose.position.z,
                     self.odom.pose.pose.orientation.x,
                     self.odom.pose.pose.orientation.y,
                     self.odom.pose.pose.orientation.z,
                     self.odom.pose.pose.orientation.w]
        with open("./src/turtlebot4_lab2/bagreader/odom.csv", "a") as f:
            for val in range(len(info_list)):
                f.write(str(info_list[val]))
                if val < len(info_list) - 1:
                    f.write(",")
            f.write("\n")


def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor
    bagreader = Bagreader()
    # pause the program execution, waits for a request to kill the node (ctrl+c)
    rclpy.spin(bagreader)
    # Explicity destroys the node
    bagreader.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()


if __name__ == '__main__':
    main()
