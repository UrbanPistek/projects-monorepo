import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Point

from turtlebot4_lab3_cmake.action import NavigationAndControl


class NavigationAndControlClient(Node):
    def __init__(self):  # initialize the client
        super().__init__('nav_and_control_client')
        self._action_client = ActionClient(self, NavigationAndControl, 'NavigationAndControl')

    def send_goal(self, x, y):
        # create a goal message
        goal_msg = NavigationAndControl.Goal()
        goal_msg.end_goal = Point()
        goal_msg.end_goal.x = x
        goal_msg.end_goal.y = y

        # send the goal message to the server. Add callbacks for feedback and for response
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        # check if the goal was accepted
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        # add callback for result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # print result and shutdown client
        reached = future.result().result.reached
        self.get_logger().info(f'Reached: ({reached.x}, {reached.y})')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        # print feedback
        goal = feedback_msg.feedback.intermediary_goal
        self.get_logger().info(f'Intermediary Goal: ({goal.x}, {goal.y})')


def main(args=None):
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    print(f"Initializing client with goal ({x}, {y})")

    rclpy.init(args=args)
    action_client = NavigationAndControlClient()
    action_client.send_goal(x, y)
    rclpy.spin(action_client)


if __name__ == '__main__':
    main()