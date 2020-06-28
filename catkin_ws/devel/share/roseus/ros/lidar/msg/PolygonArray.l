;; Auto-generated. Do not edit!


(when (boundp 'lidar::PolygonArray)
  (if (not (find-package "LIDAR"))
    (make-package "LIDAR"))
  (shadow 'PolygonArray (find-package "LIDAR")))
(unless (find-package "LIDAR::POLYGONARRAY")
  (make-package "LIDAR::POLYGONARRAY"))

(in-package "ROS")
;;//! \htmlinclude PolygonArray.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass lidar::PolygonArray
  :super ros::object
  :slots (_header _polygons _labels _likelihood ))

(defmethod lidar::PolygonArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:polygons __polygons) (let (r) (dotimes (i 0) (push (instance geometry_msgs::PolygonStamped :init) r)) r))
    ((:labels __labels) (make-array 0 :initial-element 0 :element-type :integer))
    ((:likelihood __likelihood) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _polygons __polygons)
   (setq _labels __labels)
   (setq _likelihood __likelihood)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:polygons
   (&rest __polygons)
   (if (keywordp (car __polygons))
       (send* _polygons __polygons)
     (progn
       (if __polygons (setq _polygons (car __polygons)))
       _polygons)))
  (:labels
   (&optional __labels)
   (if __labels (setq _labels __labels)) _labels)
  (:likelihood
   (&optional __likelihood)
   (if __likelihood (setq _likelihood __likelihood)) _likelihood)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; geometry_msgs/PolygonStamped[] _polygons
    (apply #'+ (send-all _polygons :serialization-length)) 4
    ;; uint32[] _labels
    (* 4    (length _labels)) 4
    ;; float32[] _likelihood
    (* 4    (length _likelihood)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; geometry_msgs/PolygonStamped[] _polygons
     (write-long (length _polygons) s)
     (dolist (elem _polygons)
       (send elem :serialize s)
       )
     ;; uint32[] _labels
     (write-long (length _labels) s)
     (dotimes (i (length _labels))
       (write-long (elt _labels i) s)
       )
     ;; float32[] _likelihood
     (write-long (length _likelihood) s)
     (dotimes (i (length _likelihood))
       (sys::poke (elt _likelihood i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; geometry_msgs/PolygonStamped[] _polygons
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _polygons (let (r) (dotimes (i n) (push (instance geometry_msgs::PolygonStamped :init) r)) r))
     (dolist (elem- _polygons)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; uint32[] _labels
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _labels (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _labels i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     ))
   ;; float32[] _likelihood
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _likelihood (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _likelihood i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get lidar::PolygonArray :md5sum-) "709b37d39871cfdbbfbd5c41cf9bc2be")
(setf (get lidar::PolygonArray :datatype-) "lidar/PolygonArray")
(setf (get lidar::PolygonArray :definition-)
      "# PolygonArray is a list of PolygonStamped.
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
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/PolygonStamped
# This represents a Polygon with reference coordinate frame and timestamp
Header header
Polygon polygon

================================================================================
MSG: geometry_msgs/Polygon
#A specification of a polygon where the first and last points are assumed to be connected
Point32[] points

================================================================================
MSG: geometry_msgs/Point32
# This contains the position of a point in free space(with 32 bits of precision).
# It is recommeded to use Point wherever possible instead of Point32.  
# 
# This recommendation is to promote interoperability.  
#
# This message is designed to take up less space when sending
# lots of points at once, as in the case of a PointCloud.  

float32 x
float32 y
float32 z
")



(provide :lidar/PolygonArray "709b37d39871cfdbbfbd5c41cf9bc2be")


