# Autonomous Mobile Robots

## Install 

Regular Install:
https://docs.ros.org/en/galactic/Installation/Ubuntu-Install-Debians.html

Using TurtleBot4 Script:
https://github.com/turtlebot/TurtleBot4Lessons/tree/main/installation

### Test Install
Terminal 1: 
```
source /opt/ros/galactic/setup.bash
ros2 run demo_nodes_cpp talker
```

Terminal 2: 
```
source /opt/ros/galactic/setup.bash
ros2 run demo_nodes_py listener
```

Check Distro: 
```
printenv ROS_DISTRO
```

## Usage
Source ROS2:
```
source /opt/ros/galactic/setup.bash
```

Using other shells such as `zsh`: 
```
source /opt/ros/galactic/setup.zsh
```

## Notes

### Record Topics to Bag File

```
rosbag record -O subset /turtle1/cmd_vel /turtle1/pose
```

## TurtleBot4
https://turtlebot.github.io/turtlebot4-user-manual/software/turtlebot4_packages.html

