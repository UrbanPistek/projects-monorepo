import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from math import pow, atan2, sqrt, acos, pi
from rclpy.qos import qos_profile_sensor_data

class Turtle(Node):


    # def __init__(self):
    #     super().__init__('move_line')
    #     self.subscriber = self.create_subscription(Odometry, '/odom', self.update_position, qos_profile_sensor_data)
    #     self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
    #     self.initialized = False

    # def callback(self, odom):
    #     if not self.initialized:
    #         theta = odom.pose.pose.orientation
    #         self.goal = [odom.pose.pose.position.x+1, ]

    
    def __init__(self):
        super().__init__('turtlebot4_move_line')

        # Each goal position to go towards
        self.idx = 0
        self.goal_positions = [(1, 1), (0, 0)]
        
        # movement patterns encoded into array as [direction, forward]
        self.movements = [['x', True, pi], ['x', False, 0]]

        self.goal_theta = [0, pi]
        self.theta_i = 0

        # Create publisher to send velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe to turtle position topic
        self.subscriber = self.create_subscription(Odometry, '/odom', self.update_position, qos_profile_sensor_data)
        
        # Get position
        self.odom = Odometry()

        # Check if initialized state
        self.initialized = False

        # Keep track of turn angle
        self.theta = 0

        # Tolerance for end goal
        self.goal_tolerance = 0.25
        self.theta_tolerance = 0.05 # 0.05 rad tolerance 

        # Velocity message
        self.vel_msg = Twist()

        # Turning Flag 
        self.turning = False

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
        self.vel_msg.linear.z = 0.0
        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0
        self.publisher.publish(self.vel_msg)

    def update_position(self, odom: Odometry):

        # Current Position
        print(f"\nCurrent Position: ({odom.pose.pose.position.x},{odom.pose.pose.position.y})")

        if not self.initialized:
            self.theta = acos(odom.pose.pose.orientation.w) * 2
            self.goal_positions = [(odom.pose.pose.position.x+1, odom.pose.pose.position.y), (odom.pose.pose.position.x, odom.pose.pose.position.y)]

            theta = acos(odom.pose.pose.orientation.w) * 2
            print(f"\nCurrent Theta: ({self.theta})")

            # Command robot to keep turning
            self.vel_msg.angular.z = 0.5

            # Check if fully turned
            if abs(self.theta) < (self.theta_tolerance):
                self.vel_msg.angular.z = 0.0
                self.turning = False
                self.initialized = True
                print(f"Initialized...")
            
            self.publisher.publish(self.vel_msg)

        else:
            # Check if in turning state
            if self.turning:
                self.theta = self.movements[self.idx][2]
                
                theta = acos(odom.pose.pose.orientation.w) * 2
                #print(f"\nCurrent Theta: ({self.theta})")

                # Command robot to keep turning
                self.vel_msg.angular.z = 0.5

                # Check if fully turned
                print("theta",theta)
                if abs(self.goal_theta[self.theta_i] - theta) < (self.theta_tolerance):
                    self.vel_msg.angular.z = 0.0
                    self.turning = False
                    self.theta_i += 1
                    self.theta_i %= 2
                
                self.publisher.publish(self.vel_msg)
            else:
                # Set goal position
                goal_pose = Pose()
                goal_pose.position.x = self.goal_positions[self.idx][0]
                goal_pose.position.y = self.goal_positions[self.idx][1]
                print(f"[{self.idx}] Goal Position: ({goal_pose.position.x},{goal_pose.position.y})")

                # Update movement pattern
                direction = self.movements[self.idx][0]
                forward = self.movements[self.idx][1]
                print(f"Movement: (dir:{direction},forward:{forward})")

                # Go straight
                self.go_straight(direction=direction, forward=forward)

                #Publish the velocity
                self.publisher.publish(self.vel_msg)

                curr_pose = odom.pose.pose.position
                dist = self.calc_distance(goal_pose, curr_pose)
                print(f"Distance to Goal: {dist}")

                # Trigger turning
                if (dist < self.goal_tolerance) and (not self.turning):
                    print(">>> Turning...")
                    
                    # Continue looping through 0-3
                    self.idx = int((self.idx + 1)%2)
                    
                    # After the loop, stops the robot
                    self.vel_msg.linear.x = 0.0
                    self.vel_msg.linear.y = 0.0
                    
                    # Force the robot to stop
                    self.publisher.publish(self.vel_msg)

                    # Enter turning mode
                    self.turning = True

    def calc_distance(self, goal_pose, pose: Pose):
        return sqrt(pow((goal_pose.position.x - pose.x), 2) +
                    pow((goal_pose.position.y - pose.y), 2))

    def go_straight(self, speed=0.2, direction='x', forward=True):
        
        vel = speed

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
