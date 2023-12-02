#!/usr/bin/env python
import rospy
import rosnode

from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from math import pow, atan2, sqrt

class Turtle:

    def __init__(self) -> None:

        # Initialize time
        self.rate = rospy.Rate(10)

        # Create publisher to send velocity commands
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=15)
        
        # Subscribe to turtle position topic
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_position)
        
        # Get position
        self.pose = Pose()

        # Tolerance for end goal
        self.goal_tolerance = 0.05

        # Velocity message
        self.vel_msg = Twist()

        # Set all to default to zero
        self.vel_msg.linear.x = 0
        self.vel_msg.linear.y = 0
        self.vel_msg.linear.z = 0
        self.vel_msg.angular.x = 0
        self.vel_msg.angular.y = 0
        self.vel_msg.angular.z = 0

        # Register shutdown callback
        rospy.on_shutdown(self.__shutdown_sequence)

        # List other running nodes
        print(f"nodes: {rosnode.get_node_names()}")

    def __shutdown_sequence(self):
        print("\nShutdown...")
        self.vel_msg.linear.x = 0
        self.vel_msg.linear.y = 0
        self.velocity_publisher.publish(self.vel_msg)

    def update_position(self, data):
        self.pose = data

    def calc_distance(self, goal_pose):
        return sqrt(pow((goal_pose.x - self.pose.x), 2) +
                    pow((goal_pose.y - self.pose.y), 2))

    def go_straight(self, speed=1, direction='x', forward=True):
        
        vel = speed
        if(forward):
            vel = abs(speed)
        else:
            vel = -abs(speed)

        if direction == 'x':
            self.vel_msg.linear.x = vel
        else:
            self.vel_msg.linear.y = vel

    def move(self):

        # Set goal position
        goal_pose = Pose()

        # Each goal position to go towards
        idx = 0
        goal_positions = [(5.54, 7.54), (7.54, 7.54), (7.54, 5.54), (5.54, 5.54)]
        
        # movement patterns encoded into array as [direction, forward]
        movements = [['y', True], ['x', True], ['y', False], ['x', False]]

        while not rospy.is_shutdown():

            # Update goal position
            goal_pose.x = goal_positions[idx][0]
            goal_pose.y = goal_positions[idx][1]
            print(f"[{idx}] Goal Position: ({goal_pose.x},{goal_pose.y})")

            # Update movement pattern
            direction = movements[idx][0]
            forward = movements[idx][1]
            print(f"Movement: (dir:{direction},forward:{forward})")

            # Loop to move the turtle in an specified distance
            # Loop until near position
            while ((self.calc_distance(goal_pose)) > self.goal_tolerance) and (not rospy.is_shutdown()):

                # Current Position
                print(f"Current Position: ({self.pose.x},{self.pose.y})")

                # Go straight
                self.go_straight(direction=direction, forward=forward)

                #Publish the velocity
                self.velocity_publisher.publish(self.vel_msg)

                # Sleep for 0.5s 
                rospy.sleep(0.5)
            
            # Continue looping through 0-3
            idx = int((idx + 1)%4)
            
            #After the loop, stops the robot
            self.vel_msg.linear.x = 0
            self.vel_msg.linear.y = 0
            
            #Force the robot to stop
            self.velocity_publisher.publish(self.vel_msg)

if __name__ == "__main__":

    # Start node
    rospy.init_node('turtle_publisher')
    
    # Create turtle object
    turtle = Turtle()

    # Start sturtle
    turtle.move()

    rospy.spin()
