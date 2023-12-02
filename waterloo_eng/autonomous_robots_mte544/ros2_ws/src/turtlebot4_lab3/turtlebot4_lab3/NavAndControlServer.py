import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point, Twist, Pose, Quaternion
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

from math import sqrt, asin, atan2, pi, cos, sin
import numpy as np
import time
import sys

from turtlebot4_lab3_cmake.action import NavigationAndControl

from turtlebot4_lab3.search import run_astar


class NavigationAndControlServer(Node):
    def __init__(self, pgm_path, yaml_path):
        super().__init__('nav_and_control_server')

        # store the pgm and yaml paths
        self.pgm_path = pgm_path
        self.yaml_path = yaml_path

        # initialize variables to hold current state
        self.position = Point()
        self.angle = 0
        self.position_initialized = False

        #initialize tolerance. When moving, the robot must be within the tolerance specified below
        self.tolerance = 0.1  # 10 cm

        # initialize the publisher, odom subscriber, and tf2 subscriber
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 20)
        self.subscriber = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile_sensor_data) 
        qos = QoSProfile(
            durability=QoSDurabilityPolicy.VOLATILE, 
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=10)
        self.subscriber = self.create_subscription(TFMessage, "/tf", self.tf2_callback, qos) 

        # initialize variables to store intermediate goals and index of next goal
        self.intermediete_goals = None
        self.i_goal = 0

        # initialize variables to store the transform between odom and map
        self.transform_initialized = False
        self.translation = None
        self.rotation = None

        # initialize values that control the robot speed
        self.linear_speed = 0.07  # 7 cm/s
        self.rotational_gain = 0.7

        #initialize a variable to store whether we're ready to move
        self.ready_to_move = False

        # initialize a variable to hold the current goal_handle
        self.goal_handle = None
        self.result = None

        # initialize a timer to be called every 0.1 seconds
        self.timer = self.create_timer(0.1, self.on_timer)

        # Initialize this node as the Navigation And Control Action Server
        self._action_server = ActionServer(
            self,
            NavigationAndControl,
            'NavigationAndControl',
            self.server_callback)

    def on_timer(self):
        if self.ready_to_move:
            goal = self.intermediete_goals[self.i_goal]  # get current goal
            goal_point = Point()
            goal_point.x = goal[0]
            goal_point.y = goal[1]

            # print(self.get_dist(self.position, goal_point))

            if self.get_dist(self.position, goal_point) <= self.tolerance:  # if reached goal
                self.i_goal += 1  # increment index of next goal

                # Publish feedback
                feedback_msg = NavigationAndControl.Feedback()
                feedback_msg.intermediary_goal = goal_point
                self.get_logger().info(f'Feedback: ({goal_point.x}, {goal_point.y}, {goal_point.z})')
                self.goal_handle.publish_feedback(feedback_msg)

            cmd = Twist()

            if self.i_goal < self.intermediete_goals.shape[0]:  # if have not reached end goal yet
                # update the goal in case i_goal has incremented
                goal_point.x = self.intermediete_goals[self.i_goal, 0]
                goal_point.y = self.intermediete_goals[self.i_goal, 1]

                # set a constant linear speed
                cmd.linear.x = self.linear_speed

                # the angular speed will be P controlled
                desired_angle = atan2(goal_point.y - self.position.y, goal_point.x - self.position.x)  # rad
                cmd.angular.z = (desired_angle - self.angle) * self.rotational_gain
            else:  # if reached end goal
                # stop moving
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

                self.get_logger().info(f'Got to goal: ({self.position.x}, {self.position.y}, {self.position.z})')
                self.goal_handle.succeed()

                # don't move till you get a new goal
                self.ready_to_move = False

            self.publisher.publish(cmd)

    async def server_callback(self, goal_handle):
        print("server called")
        self.goal_handle = goal_handle  # store the goal handle

        end_goal = goal_handle.request.end_goal

        if not self.position_initialized:
            self.get_logger().warn(f'position not initialized. Returning empty')
            goal_handle.abort()
            return NavigationAndControl.Result()

        # Get intermediary goals through A star
        print(f"start: {self.position.x}, {self.position.y}")
        print(f"goal: {end_goal.x}, {end_goal.y}")
        self.intermediete_goals = run_astar(self.pgm_path, self.yaml_path, (self.position.x, self.position.y), (end_goal.x, end_goal.y))
        # self.intermediete_goals = np.random.random((100, 2))  # for testing

        self.ready_to_move = True
        self.get_logger().info(f'Executing goal of ({end_goal.x}, {end_goal.y}, {end_goal.z})')

        self.result = NavigationAndControl.Result()

        print("spinning")
        while self.ready_to_move:
            time.sleep(1)
            print("spun")
        print("returning")
        self.result.reached = self.position
        return self.result

    def get_dist(self, p1: Point, p2: Point) -> float:
        return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def get_yaw(self, q) -> float:
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = q.x
        y = q.y
        z = q.z
        w = q.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)
     
        return yaw_z # in radians

    def odom_callback(self, msg):
        # updates the current position based on the latest msg in the /tf topic
        if self.transform_initialized:
            # gets the position in the odom frame
            odom_x = msg.pose.pose.position.x
            odom_y = msg.pose.pose.position.y

            # transforms position to the global frame
            self.position.x = cos(self.rotation)*odom_x - sin(self.rotation)*odom_y + self.translation.x
            self.position.y = -sin(self.rotation)*odom_x + cos(self.rotation)*odom_y + self.translation.y

            # transforms angle to the global frame
            self.angle = self.get_yaw(msg.pose.pose.orientation) + self.rotation

            self.position_initialized = True

            print(f"map: ({self.position.x}, {self.position.y}), {self.angle}")
            print(f"odom: ({odom_x}, {odom_y}), {self.get_yaw(msg.pose.pose.orientation)}")
        else: 
            print("transform not initialized")

    def tf2_callback(self, msg):
        # looks for the transform between the map frame and the odom frame
        if not self.transform_initialized:
            for transform in msg.transforms:  # iterates through all transforms in the message
                print(f"parent: '{transform.header.frame_id}', child: '{transform.child_frame_id}'")
                if transform.header.frame_id == "map" and transform.child_frame_id == "odom":  # if the required transform
                    self.get_logger().info(f'Found transform between map and odom')

                    # store transfrom between map and odom
                    self.translation = transform.transform.translation
                    self.rotation = self.get_yaw(transform.transform.rotation)

                    print(f"transform: x={self.translation.x}, y={self.translation.y}, theta={self.rotation}")

                    # indicate that transform found and break
                    self.transform_initialized = True
                    break



def main(args=None):
    rclpy.init(args=args)
    nav_and_control_action_server = NavigationAndControlServer(sys.argv[1], sys.argv[2])

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(nav_and_control_action_server, executor=executor)


if __name__ == '__main__':
    main()