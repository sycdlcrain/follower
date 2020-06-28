# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from lidar/PolygonArray.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg
import std_msgs.msg

class PolygonArray(genpy.Message):
  _md5sum = "709b37d39871cfdbbfbd5c41cf9bc2be"
  _type = "lidar/PolygonArray"
  _has_header = True #flag to mark the presence of a Header object
  _full_text = """# PolygonArray is a list of PolygonStamped.
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
float32 z"""
  __slots__ = ['header','polygons','labels','likelihood']
  _slot_types = ['std_msgs/Header','geometry_msgs/PolygonStamped[]','uint32[]','float32[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,polygons,labels,likelihood

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(PolygonArray, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.polygons is None:
        self.polygons = []
      if self.labels is None:
        self.labels = []
      if self.likelihood is None:
        self.likelihood = []
    else:
      self.header = std_msgs.msg.Header()
      self.polygons = []
      self.labels = []
      self.likelihood = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.polygons)
      buff.write(_struct_I.pack(length))
      for val1 in self.polygons:
        _v1 = val1.header
        buff.write(_get_struct_I().pack(_v1.seq))
        _v2 = _v1.stamp
        _x = _v2
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v1.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _v3 = val1.polygon
        length = len(_v3.points)
        buff.write(_struct_I.pack(length))
        for val3 in _v3.points:
          _x = val3
          buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
      length = len(self.labels)
      buff.write(_struct_I.pack(length))
      pattern = '<%sI'%length
      buff.write(struct.pack(pattern, *self.labels))
      length = len(self.likelihood)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.likelihood))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.polygons is None:
        self.polygons = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.polygons = []
      for i in range(0, length):
        val1 = geometry_msgs.msg.PolygonStamped()
        _v4 = val1.header
        start = end
        end += 4
        (_v4.seq,) = _get_struct_I().unpack(str[start:end])
        _v5 = _v4.stamp
        _x = _v5
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v4.frame_id = str[start:end].decode('utf-8')
        else:
          _v4.frame_id = str[start:end]
        _v6 = val1.polygon
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        _v6.points = []
        for i in range(0, length):
          val3 = geometry_msgs.msg.Point32()
          _x = val3
          start = end
          end += 12
          (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
          _v6.points.append(val3)
        self.polygons.append(val1)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sI'%length
      start = end
      end += struct.calcsize(pattern)
      self.labels = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.likelihood = struct.unpack(pattern, str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.polygons)
      buff.write(_struct_I.pack(length))
      for val1 in self.polygons:
        _v7 = val1.header
        buff.write(_get_struct_I().pack(_v7.seq))
        _v8 = _v7.stamp
        _x = _v8
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v7.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _v9 = val1.polygon
        length = len(_v9.points)
        buff.write(_struct_I.pack(length))
        for val3 in _v9.points:
          _x = val3
          buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
      length = len(self.labels)
      buff.write(_struct_I.pack(length))
      pattern = '<%sI'%length
      buff.write(self.labels.tostring())
      length = len(self.likelihood)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.likelihood.tostring())
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.polygons is None:
        self.polygons = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.polygons = []
      for i in range(0, length):
        val1 = geometry_msgs.msg.PolygonStamped()
        _v10 = val1.header
        start = end
        end += 4
        (_v10.seq,) = _get_struct_I().unpack(str[start:end])
        _v11 = _v10.stamp
        _x = _v11
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v10.frame_id = str[start:end].decode('utf-8')
        else:
          _v10.frame_id = str[start:end]
        _v12 = val1.polygon
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        _v12.points = []
        for i in range(0, length):
          val3 = geometry_msgs.msg.Point32()
          _x = val3
          start = end
          end += 12
          (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
          _v12.points.append(val3)
        self.polygons.append(val1)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sI'%length
      start = end
      end += struct.calcsize(pattern)
      self.labels = numpy.frombuffer(str[start:end], dtype=numpy.uint32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.likelihood = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_3f = None
def _get_struct_3f():
    global _struct_3f
    if _struct_3f is None:
        _struct_3f = struct.Struct("<3f")
    return _struct_3f
