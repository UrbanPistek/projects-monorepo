from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python import get_package_share_directory
import launch 

from launch.actions import ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return launch.LaunchDescription([
      launch.actions.ExecuteProcess(
      	cmd = ['ros2', 'bag', 'play',  '/home/cmele/Map1/testPoint'],
      	#output = 'screen'
      ),
        Node(
            package='bagreader',
            executable='bagrun',
            output = 'screen'),
    ])

