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

from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, PoseArray, Polygon, Point, Pose2D, Pose
from sensor_msgs.msg import PointCloud2, Joy

import matplotlib.pyplot as plt
# ~ import tensorflow as tf
import RPi.GPIO as GPIO
import time

from params import *

VIS_PATH = 0
VIS_STEER = 0

def get_corners(state):
	corners = []
	x = state[0]
	y = state[1]
	theta = state[2]
	pts = np.asarray([[-width/2.0, -height/2.0],
					  [-width/2.0, height/2.0],
					  [width/2.0, height/2.0],
					  [width/2.0, -height/2.0]])
	R = np.asarray([[np.cos(theta), -np.sin(theta), 0],
					[np.sin(theta), np.cos(theta),  0],
					[0,   0,    1]])
	rot_pts = np.dot(R, np.hstack((pts,np.ones((4,1)))).transpose()).transpose()
	rot_pts[:,0]+=x
	rot_pts[:,1]+=y
	corners = rot_pts[:,:2]
	return corners

def close_poly(corners):
	return np.vstack((corners,corners[0,:]))

def pure_pursuit(goal):
	
	# CURVATURE CALCULATION
	# x = forward, y is lateral
	# o			o	y
	# |---------|	|
	# |			|   ------x
	# |---------|	
	# o			o	
	#				

	# # CALCULATE PURE PURSUIT
	x = goal[0]
	y = goal[1]
	l2 = (x**2+y**2)
	twox = 2*y
	r = (l2+0.0)/twox

	# # if x is negative, we need to go in the other direction
	s = np.sign(y)
	# print("l2", l2,"2x",twox, "r", r, "signed extent", s*width/2)

	width = 0.6 #1.2
	height = 0.6
	# # if y is neg, the same arc will get you there as if it were positive
	angle = np.arctan2(s*width/2, np.abs(r))
	print("angle", angle)

	if VIS_STEER:
		plt.clf()
		
		plt.plot(0,0,'.k')
		plt.plot(goal[0],goal[1],'or')
		plt.plot([0,np.cos(angle)],[0,np.sin(angle)],'b')

		plt.axis([-2, 10, -2, 10])

		plt.show(block=False)
		plt.pause(0.0000000001)
	# speed is a function of target speed, target distance, and curvature
	# ~ goal_dist = np.sqrt(np.sum((spline[-1,:2] - ego[:2])**2))
	# ~ q = 1.0

	# action = [q/goal_dist, angle]
	# action = [0.2, angle]
	return angle
	
def get_accel(current_v, goal, goal_speed, max_speed, dt, max_accel=1.0):
	# ~ distance_to_goal = np.sqrt(np.sum((state[0:2]-goal[0:2])**2))
	distance_to_goal = np.sqrt(np.sum((goal[0:2])**2))

	# follower needs to be behind
	eps = 0.3
	goal_speed = np.maximum(goal_speed-eps,0.0)

	if goal_speed>max_speed:
		tv = max_speed
	else:
		# faster the further ego is from goal

		c = 5.0 # max distance
		d = np.minimum(distance_to_goal, c)
		ratio = d/c
		# print("dist", d, ratio)
		tv = (1-ratio)*goal_speed + ratio*max_speed
	

	v_error = tv - current_v
	accel = np.minimum(max_accel, v_error/dt)
	# print("1--", max_accel, v_error/dt, accel)
	accel = np.maximum(-max_accel, accel)
	# print("2--", -max_accel, accel, accel)
	# print("tv", tv, "err", v_error)
	# print(ratio, "v", state[3], goal_speed, "a", accel)
	return accel, tv
##################################################################################


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
		self.target = np.asarray([1.0,0])
		self.target_vel = 0.0 # np.asarray([0.0, 0.0, 0.0])

		self.timer = time.time()
		self.control_allowed = 0
		
		# PUBLISHER
		# self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		# ~ self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)

		#SUBSCRIBER
		# ~ rospy.Subscriber('/low_level/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, self.parse_command)
		# ~ rospy.Subscriber('/path_to_target', Marker, self.read_path)
		rospy.Subscriber('/tracked_target', Marker, self.process_target)
		rospy.Subscriber('/tracked_vel', Point, self.process_vel)
		self.joypad_sub = rospy.Subscriber('/joy', Joy, self.process_joypad)
		
		rospy.sleep(0.01) 
		
		rospy.loginfo("Waiting for topics...")
		try:
			# ~ rospy.wait_for_message('/low_level/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, timeout=10.0)
			rospy.wait_for_message('/tracked_target', Marker, timeout=10.0)
		except rospy.ROSException as e:
			rospy.logerr("Timeout while waiting for topics!")
			raise e
		print("Node started")
		
		while not rospy.is_shutdown():
			self.r.sleep()
			
			if (time.time()-self.timer)>0.5:
				self.control_allowed = 0
				
		
			self.get_commands()
			self.openloop_steer()
			self.openloop_drive()
			# ~ self.pub.publish(self.data) # numpy in ros only handles 1d arrays
			

	def shutdown(self):
		rospy.loginfo("Stop control_node")
		GPIO.cleanup()
		# rospy.sleep(1)
		return 0

	def process_joypad(self, data):
	
		shoulder = data.buttons[4]
		a_button = data.buttons[0] # X on supernintendo
		
		# ~ print("joy:", shoulder, a_button)
		if shoulder==1: # and a_button==1:
			print("ALLOWING control")
			self.control_allowed = 1
			self.timer = time.time()
		else:
			self.control_allowed = 0

	def process_target(self, data):
		# ~ print(data.drive.steering_angle)
		self.target = np.asarray([data.pose.position.x, data.pose.position.y, data.pose.position.z])
	
	def process_vel(self, data):
		# ~ print(data.drive.steering_angle)
		local_vel = np.sqrt(np.sum(np.asarray([data.x, data.y, data.z])**2))
		self.target_vel	= local_vel + self.vel

	def get_commands(self):
		# ~ state = update_pose(state, action, dt)
		# ~ person = update_person(person, dt)
		# ~ goal = find_goal(person) # offset position
		goal = self.target[:2] - [1.0, 0.0]

		# BOUND ACTION
		# ~ distance_to_goal = np.sqrt(np.sum((state[0:2]-goal[0:2])**2)) # FIX TO BE NON-EUCLIDEAN 
		max_speed = 2.0
		dt = 0.1
		# ~ target_speed = np.mean([self.target_vel, max_speed])
		# ~ timesteps = 50 
		# ~ x, y = get_spline(state[0], goal[0], state[1], goal[1], state[2], goal[2], steps=timesteps) # not necessarily feasible
		# ~ traj = np.vstack([x,y]).transpose()

		self.steer = pure_pursuit(goal)
		accel, self.vel = get_accel(self.vel, goal, self.target_vel, max_speed, dt)
		# ~ action = [accel, self.steer]
		# ~ print("final", self.steer, self.vel)

		# VISUALIZE 
		if VIS_PATH:
			plt.clf()
			# plt.plot(trajectory[:,0], trajectory[:,1],'.')
			corners = close_poly(get_corners(state+0.0))
			plt.plot(traj[:,0],traj[:,1],'.k')
			plt.plot(person[0],person[1],'ob')
			plt.plot(goal[0],goal[1],'or')
			plt.plot(corners[:,0],corners[:,1],'k')

			opt_trajectory = get_trajectory(state, action, 50)
			plt.plot(opt_trajectory[:,0], opt_trajectory[:,1],'xr')
			corners = close_poly(get_corners(opt_trajectory[-1,:]))
			plt.plot(corners[:,0],corners[:,1],'r')
			# plt.axis("equal")
			plt.axis([0, 20, 0, 20])

			plt.show(block=False)
			plt.pause(0.0000000001)
		
		

	def openloop_steer(self):
		# MAP ANGLE(-.34 to .34) to PWM(0 30)
		angle_range = .34-(-.34)
		angle_min = -.34
		angle_ratio = (self.steer-angle_min)/angle_range
		
		# 18 - 30?
		PWM_min = 10.0
		PWM_range = 16.0-PWM_min
		
		mapped_val = angle_ratio*PWM_range + PWM_min
		
		print("angle:",self.steer, angle_ratio, " mapped to:", mapped_val)
		self.steer_pwm.ChangeDutyCycle(int(mapped_val))
		# ~ print("int:",int(mapped_val))

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
		
		print("drive:",speed_ratio, " mapped to:", mapped_val)
		if self.control_allowed:
			self.motor_pwm.ChangeDutyCycle(mapped_val)
		else:
			self.motor_pwm.ChangeDutyCycle(0.0)
		# ~ print("int:",int(mapped_val))
		


		
if __name__ == '__main__':
	try:
		control_node()
	except rospy.ROSInterruptException:
		pass


