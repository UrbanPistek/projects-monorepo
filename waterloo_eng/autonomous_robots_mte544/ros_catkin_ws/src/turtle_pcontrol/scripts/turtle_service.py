#!/usr/bin/env python
from turtle_pcontrol.srv import CalcVel, CalcVelResponse
import rospy

from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from math import pow, atan2, sqrt

class Turtle:

    def __init__(self) -> None:

        # Create publisher to send velocity commands
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=15)
        
        # Subscribe to turtle position topic
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_position)
        
        # Get position
        self.pose = Pose()

        # Tolerance for end goal
        self.goal_tolerance = 0.05

        self.linear_constant = 1
        self.angular_constant = 5

    def update_position(self, data):
        self.pose = data

    def calc_distance(self, goal_pose):
        return sqrt(pow((goal_pose.x - self.pose.x), 2) +
                    pow((goal_pose.y - self.pose.y), 2))

    def linear_vel(self, goal_pose):
        return self.linear_constant * self.calc_distance(goal_pose)

    def angular_vel(self, goal_pose):

        theta = atan2(goal_pose.y - self.pose.y, goal_pose.x - self.pose.x)
        return self.angular_constant * (theta - self.pose.theta)

    def go_to_position(self, req):
        print(f"Goal position: ({req.x}, {req.y})")
        print(f"Starting Position: ({self.pose.x}, {self.pose.y})")

        # Set goal position
        goal_pose = Pose()
        goal_pose.x = float(req.x)
        goal_pose.y = float(req.y)

        # Great twist message
        vel_msg = Twist()

        # Set all values that will remain constant
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0

        # Loop until near position
        while (self.calc_distance(goal_pose)) > self.goal_tolerance:

            # Current Position
            print(f"Current Position: ({self.pose.x},{self.pose.y})")

            # Linear velocity in the x-axis.
            vel_msg.linear.x = self.linear_vel(goal_pose)

            # Angular velocity in the z-axis.
            vel_msg.angular.z = self.angular_vel(goal_pose)

            # Publishing our vel_msg
            self.velocity_publisher.publish(vel_msg)

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping our robot after the movement is over.
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

        # Indicate position has been reached
        return CalcVelResponse(True)

    def calc_vel_server(self):

        # Start service
        rospy.init_node('calc_vel_server')
                
        # Initialize time
        self.rate = rospy.Rate(10)

        # Create service
        s = rospy.Service('calc_vel', CalcVel, self.go_to_position)
        print("Ready to calculate velocity...")
        
        # loops forever
        rospy.spin()

if __name__ == "__main__":
    
    # Create turtle object
    turtle = Turtle()

    # Start server
    turtle.calc_vel_server()
