cmake_minimum_required(VERSION 2.8.3)
project(control)

# Add CMake modules folder BEFORE we try to find libraries
# include(${CMAKE_CURRENT_SOURCE_DIR}/../../common_modules/CMakeModules/CMakeLists.txt)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  genmsg
  message_generation
)


## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

## Do I need this?
##add_service_files(
##  DIRECTORY srv
##  FILES
##  GetLanelet.srv
##)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS 
  	rospy 
  	std_msgs 
  	message_runtime
  	#    razor_imu_9dof
    tf
    tf2_ros
    urg_node
    joy
    rosbag
    rostopic
    rviz
    # specific to racecar-v1
    #pointgrey_camera_driver
    #px4flow
    #pwm_sysfs_driver
    # specific to racecar-v2
    # mapping
    gmapping
    hector_mapping
    robot_pose_ekf
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(control src/publish_subscribe_placeholder.cpp)

target_link_libraries(control
   ${catkin_LIBRARIES}
#    ${honda_libs_LIBRARIES}
#    ${simulator_gazebo_LIBRARIES}
#    ${PCL_LIBRARIES}
#    ${zenrin_map_LIBRARIES}
)
add_dependencies(control ${catkin_EXPORTED_TARGETS})

# add_executable(turtlebot_teleop_joy src/turtlebot_joy.cpp)
# target_link_libraries(turtlebot_teleop_joy ${catkin_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
#install(PROGRAMS
#  scripts/dummy_node
#  scripts/gonogo_prediction
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

## Mark executables and/or libraries for installation
#install(TARGETS vehicle_tracking ttc_calculation
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

## Mark cpp header files for installation
#install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
#)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_intersection.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
