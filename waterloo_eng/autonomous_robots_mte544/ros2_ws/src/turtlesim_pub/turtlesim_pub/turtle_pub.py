import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from math import pow, atan2, sqrt

class Turtle(Node):
    
    def __init__(self):
        super().__init__('turtle_pub')

        # Each goal position to go towards
        self.idx = 0
        self.goal_positions = [(5.54, 7.54), (7.54, 7.54), (7.54, 5.54), (5.54, 5.54)]
        
        # movement patterns encoded into array as [direction, forward]
        self.movements = [['y', True], ['x', True], ['y', False], ['x', False]]

        # Create publisher to send velocity commands
        self.velocity_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Subscribe to turtle position topic
        self.pose_subscriber = self.create_subscription(Pose, '/turtle1/pose', self.update_position, 10)
        
        # Get position
        self.pose = Pose()

        # Tolerance for end goal
        self.goal_tolerance = 0.125

        # Velocity message
        self.vel_msg = Twist()

        # Set all to default to zero
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0
        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0

    def shutdown_sequence(self):
        print("\nShutdown...")
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.velocity_publisher.publish(self.vel_msg)

    def update_position(self, pose: Pose):

        # Current Position
        print(f"\nCurrent Position: ({pose.x},{pose.y})")

        # Set goal position
        goal_pose = Pose()
        goal_pose.x = self.goal_positions[self.idx][0]
        goal_pose.y = self.goal_positions[self.idx][1]
        print(f"[{self.idx}] Goal Position: ({goal_pose.x},{goal_pose.y})")

        # Update movement pattern
        direction = self.movements[self.idx][0]
        forward = self.movements[self.idx][1]
        print(f"Movement: (dir:{direction},forward:{forward})")

        # Go straight
        self.go_straight(direction=direction, forward=forward)

        #Publish the velocity
        self.velocity_publisher.publish(self.vel_msg)

        dist = self.calc_distance(goal_pose, pose)
        print(f"Distance to Goal: {dist}")

        if (dist < self.goal_tolerance):
            print(">>> Turning...")
            # Continue looping through 0-3
            self.idx = int((self.idx + 1)%4)
            
            # After the loop, stops the robot
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.0
            
            # Force the robot to stop
            self.velocity_publisher.publish(self.vel_msg)

    def calc_distance(self, goal_pose, pose: Pose):
        return sqrt(pow((goal_pose.x - pose.x), 2) +
                    pow((goal_pose.y - pose.y), 2))

    def go_straight(self, speed=1, direction='x', forward=True):
        
        vel = speed
        if(forward):
            vel = abs(speed)
        else:
            vel = -abs(speed)

        if direction == 'x':
            self.vel_msg.linear.x = float(vel)
        else:
            self.vel_msg.linear.y = float(vel)

def main():
    rclpy.init()
    node = Turtle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.shutdown_sequence()
    node.destroy_node()

    rclpy.shutdown()
