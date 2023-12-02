cmake_minimum_required(VERSION 3.0.2)
project(turtle_pub)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES beginner_tutorials
   CATKIN_DEPENDS message_runtime roscpp rospy std_msgs 
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
	scripts/turtle_publisher.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
