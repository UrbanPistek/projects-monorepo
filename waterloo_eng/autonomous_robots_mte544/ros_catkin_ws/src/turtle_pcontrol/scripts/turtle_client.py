#!/usr/bin/env python

import rospy
from turtle_pcontrol.srv import CalcVel

def move_turtle():

    goals = [[1, 1], [1, 2], [2, 2], [2, 1]]
    
    # Wait for service
    rospy.wait_for_service('calc_vel')

    for goal in goals: 
        x = goal[0]
        y = goal[1]
        print(f"Moving to ({x}, {y})")

        # Trigger service
        try:
            send_goal_position = rospy.ServiceProxy('calc_vel', CalcVel)
            resp = send_goal_position(x, y)
            print(f"Service: {resp}")
        
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

if __name__ == "__main__":
    
    move_turtle()
