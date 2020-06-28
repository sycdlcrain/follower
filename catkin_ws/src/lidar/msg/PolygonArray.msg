# PolygonArray is a list of PolygonStamped.
# Don't forget to turn on message generation: http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv
# In package.xml:
# <build_depend>message_generation</build_depend>
# <exec_depend>message_runtime</exec_depend>
# In CMakeLists.txt:
#1. find_package(catkin REQUIRED COMPONENTS
#   ...
#   message_generation
# )
#2. catkin_package(
#  ...
#  CATKIN_DEPENDS message_runtime ...
#  ...)
#3. add_message_files(
#  FILES
#  Num.msg
# )
#4. generate_messages(
#   DEPENDENCIES
#   std_msgs
# )  

Header header
geometry_msgs/PolygonStamped[] polygons
uint32[] labels
float32[] likelihood