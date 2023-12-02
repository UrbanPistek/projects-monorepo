#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovariance
from nav_msgs.msg import Odometry
import math
from rclpy.qos import qos_profile_sensor_data


class CircleControllerNode(Node):
    def __init__(self):
        super().__init__("circle_controller_node")
        self.center_pose = PoseWithCovariance()
        self.initialized = False

        self.publisher = self.create_publisher(Twist, "/cmd_vel", 20)
        self.subscriber = self.create_subscription(Odometry, "/odom", self.callback_controller, qos_profile_sensor_data)
        self.get_logger().info("Circle controller has initialized")

    def callback_controller(self, odom):
        print("foo")
        if not self.initialized:
            self.center_pose = odom.pose
            self.center_pose.pose.position.x -= 1
            self.initialized = True

        else:
            twist = Twist()
            twist.linear.x = 0.5
            twist.angular.z = 1.0
            print("test")
            self.publisher.publish(twist)


def main(args = None):
    rclpy.init(args=args)
    node = CircleControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
    print('Hi from lab1_circle.')


if __name__ == '__main__':
    main()
