#!/usr/bin/env python

# This code poses as the rl_state being published by sensor data
# Author = David Isele


import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#from DeepRL.dqn.SumoCarMDP import *
from honda_msgs.msg import Object
from honda_msgs.msg import ObjectVector
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from ackermann_msgs.msg import AckermannDriveStamped

import matplotlib.pyplot as plt
import tensorflow as tf


class control_node():
	def __init__(self):

		## SETUP ROS NODES
		rospy.init_node('control_node', anonymous=False)
		rospy.loginfo("To stop vecs_to_gonogo CTRL + C")  
		rospy.on_shutdown(self.shutdown)
		self.r = rospy.Rate(10) # 10hz

		self.steer = 0.0
		self.vel = 0.0

		# PUBLISHER
		# self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)

		#SUBSCRIBER
		rospy.Subscriber('/low_level/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, self.parse_command)
		
		rospy.sleep(0.01) 
		
		while not rospy.is_shutdown():
			self.r.sleep()
			# ~ self.pub.publish(self.data) # numpy in ros only handles 1d arrays


	def shutdown(self):
		rospy.loginfo("Stop control_node")
		# rospy.sleep(1)
		return 0

	def parse_command(self, data):
		print(data.drive.steering_angle)


if __name__ == '__main__':
	try:
		control_node()
	except rospy.ROSInterruptException:
		pass


