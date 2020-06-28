#!/usr/bin/env python

# This code poses as the rl_state being published by sensor data
# Author = David Isele


import rospy
import numpy as np
# ~ from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#from DeepRL.dqn.SumoCarMDP import *
# ~ from honda_msgs.msg import Object
# ~ from honda_msgs.msg import ObjectVector
# ~ from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from ackermann_msgs.msg import AckermannDriveStamped

import matplotlib.pyplot as plt
# ~ import tensorflow as tf
import RPi.GPIO as GPIO

from params import *

class control_node():
	def __init__(self):

		## SETUP ROS NODES
		rospy.init_node('control_node', anonymous=False)
		rospy.loginfo("To stop vecs_to_gonogo CTRL + C")  
		rospy.on_shutdown(self.shutdown)
		self.r = rospy.Rate(10) # 10hz

		# SETUP PIN OUTPUTS
		GPIO.setwarnings(False)
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(STEERING_PIN, GPIO.OUT)
		self.steer_pwm = GPIO.PWM(STEERING_PIN , STEERING_FREQ)
		self.steer_pwm.start(0)
		
		GPIO.setup(MOTOR_PIN, GPIO.OUT)
		self.motor_pwm = GPIO.PWM(MOTOR_PIN, MOTOR_FREQ)
		self.motor_pwm.start(0)
		GPIO.setup(MOTOR_D1, GPIO.OUT)
		GPIO.setup(MOTOR_D2, GPIO.OUT)

		self.steer = 0.0
		self.vel = 0.0

		# PUBLISHER
		# self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		# ~ self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)

		#SUBSCRIBER
		rospy.Subscriber('/low_level/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, self.parse_command)
		
		rospy.sleep(0.01) 
		
		while not rospy.is_shutdown():
			self.r.sleep()
			self.openloop_steer()
			self.openloop_drive()
			# ~ self.pub.publish(self.data) # numpy in ros only handles 1d arrays
			

	def shutdown(self):
		rospy.loginfo("Stop control_node")
		GPIO.cleanup()
		# rospy.sleep(1)
		return 0

	def parse_command(self, data):
		# ~ print(data.drive.steering_angle)
		self.steer = data.drive.steering_angle
		self.vel = data.drive.speed
		

	def openloop_steer(self):
		# MAP ANGLE(-.34 to .34) to PWM(0 30)
		angle_range = .34-(-.34)
		angle_min = -.34
		angle_ratio = (self.steer-angle_min)/angle_range
		
		PWM_min = 10.0
		PWM_range = 16.0-PWM_min
		
		mapped_val = angle_ratio*PWM_range + PWM_min
		
		print("angle:",self.steer, angle_ratio, " mapped to:", mapped_val)
		self.steer_pwm.ChangeDutyCycle(int(mapped_val))
		print("int:",int(mapped_val))

	def openloop_drive(self):
		
		if self.vel>0:
			GPIO.output(MOTOR_D1, False)
			GPIO.output(MOTOR_D2, True)    
		else:
			GPIO.output(MOTOR_D1, True)
			GPIO.output(MOTOR_D2, False) 


		# MAP ANGLE(-.34 to .34) to PWM(0 30)
		speed_min = 0.0
		speed_range = 2.0-speed_min
		
		speed_ratio = (np.abs(self.vel)-speed_min)/speed_range
		
		PWM_min = 0.0
		PWM_range = 100.0-PWM_min
		
		mapped_val = speed_ratio*PWM_range + PWM_min
		
		print("angle:",speed_ratio, " mapped to:", mapped_val)
		self.motor_pwm.ChangeDutyCycle(mapped_val)
		print("int:",int(mapped_val))
		
if __name__ == '__main__':
	try:
		control_node()
	except rospy.ROSInterruptException:
		pass


