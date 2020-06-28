#!/usr/bin/env python

# track target
# Author = David Isele


import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#from DeepRL.dqn.SumoCarMDP import *
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, PoseArray, Polygon, Point, Pose2D, Pose, Twist
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, Joy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection

import matplotlib.pyplot as plt
#import tensorflow as tf

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from copy import deepcopy
import time

VIS_CLUSTER = 0
VIS_HULL = 0

def default_marker(x=0,y=0,z=0):
	m = Marker()
	m.header.frame_id = "laser"
	m.type = 2 # 4 line strip, 6 cube list # car.type
	m.scale.x = .30
	m.scale.y = .30
	m.scale.z = .30
	m.pose.position.x=x
	m.pose.position.y=y
	m.pose.position.z=z
	m.pose.orientation.x=0
	m.pose.orientation.y=0
	m.pose.orientation.z=0
	m.pose.orientation.w=1
	m.color.r = 0.0
	m.color.g = 0.3
	m.color.b = 0.7
	m.color.a = 1.0

	return m
	

class target_tracker():
	def __init__(self):

		## SETUP ROS NODES
		rospy.init_node('target_tracker', anonymous=False)
		rospy.loginfo("To stop target_tracker CTRL + C")  
		rospy.on_shutdown(self.shutdown)
		self.r = rospy.Rate(10) # 10hz

		self.coms = 0.0
		self.vel = [0.0, 0.0]
		
		self.new_pose = np.asarray([1,0])
		self.target_pose = np.asarray([1,0])
		
		self.dt = 0.1 
		# initial parameters
		self.measurement_sig = .3 	# measurement uncertainty
		self.motion_sig = .5		# motion uncertainty
		self.target_pose  = np.asarray([1,0])
		self.sig = 10000.
		self.vel_sig = 10000.
		
		#SUBSCRIBER
		self.new_target_sub = rospy.Subscriber('/new_target', Marker, self.process_new_target)
		self.hulls_sub = rospy.Subscriber('/convex_hulls', MarkerArray, self.process_hulls)
		self.joypad_sub = rospy.Subscriber('/joy', Joy, self.process_joypad)
		# target pose from filter
		
		# PUBLISHER
		# self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		# ~ self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)
		# ~ self.pub_hulls = rospy.Publisher('/convex_hulls', MarkerArray, queue_size=1)
		# ~ self.pub_tracked_target = rospy.Publisher('/tracked_target', Point, queue_size=1)
		self.pub_tracked = rospy.Publisher('/tracked_target', Marker, queue_size=1)	
		self.pub_tracked_vel = rospy.Publisher('/tracked_vel', Point, queue_size=1)		
		
		rospy.sleep(0.01) 
		
		rospy.loginfo("Waiting for topics...")
		try:
			rospy.wait_for_message('/new_target', Marker, timeout=10.0)
			rospy.wait_for_message('/convex_hulls', MarkerArray, timeout=10.0)
			rospy.wait_for_message('/joy', Joy, timeout=10.0)
		except rospy.ROSException as e:
			rospy.logerr("Timeout while waiting for topics!")
			raise e
		print("Node started")
		
		self.prev_measure = deepcopy(self.target_measurement)
		rospy.sleep(0.1)
		
		while not rospy.is_shutdown():
			self.r.sleep()
			
			# GET MEASUREMENT
			self.start = time.time()
			self.new_measure = deepcopy(self.target_measurement)			
			self.update_pose()
			
			# PUBLISH ESTIMATE
			m = default_marker(x=self.target_pose[0], y=self.target_pose[1],z=0)
			self.pub_tracked.publish(m) 
			v = Point(x=self.vel[0],y=self.vel[1],z=0)
			self.pub_tracked_vel.publish(v) 
			
			if VIS_HULL:
				fig = plt.figure(1)
				plt.clf()
				plt.plot(points[:,0], points[:,1], 'o')
				for n in range(n_clusters):
					hull = hulls[n]
					pts = points[labels==n,:]
					for simplex in hull.simplices:
						if n==self.cluster_id:
							plt.plot( hull.points[simplex, 0],  hull.points[simplex, 1], 'r-')
						else:
							plt.plot( hull.points[simplex, 0],  hull.points[simplex, 1], 'k-')
				plt.axis([-2,8,-4,4])
				plt.show(block=False)
				plt.pause(0.0000000001)	
			
			if time.time()-self.start>self.dt:
				print("too slow", time.time()-self.start, "seconds")
			while time.time()-self.start<self.dt:
				pass

	def shutdown(self):
		rospy.loginfo("Stop target_tracker")
		# rospy.sleep(1)
		return 0

	def process_hulls(self,data):

		self.coms = []
		for h in range(len(data.markers)):
			pt = data.markers[h].points[0]
			self.coms.append(  [pt.x, pt.y]  )
			
		dists = np.sum((np.vstack(self.coms)-self.target_pose)**2,1)
		self.cluster_id = np.argmin(dists)
		
		self.target_measurement = np.asarray(self.coms[self.cluster_id])


	def process_new_target(self, data):
		self.new_pose = np.asarray([data.pose.position.x, data.pose.position.y])

		
	def process_joypad(self, data):
	
		shoulder = data.buttons[4]
		a_button = data.buttons[0]
		
		# ~ print("joy:", shoulder, a_button)
		if shoulder==1 and a_button==1:
			print("RESETTING target")
			self.reset_target()


	def reset_target(self):
		self.target_pose = self.new_pose + 0.0

	def update_pose(self):
		#KALMAN FILTER
		self.target_pose, self.sig = kf_update(self.target_pose, self.sig, self.new_measure, self.measurement_sig)
		# ~ print('Update: [{}, {}]'.format(mu, sig))
		
		# CONSTANT VELOCITY MOTION (am i not filtering this?)
		new_vel = (self.new_measure - self.prev_measure)/self.dt
		vx, self.vel_sig = kf_update(self.vel[0], self.vel_sig, new_vel[0], self.motion_sig)
		vy, self.vel_sig = kf_update(self.vel[1], self.vel_sig, new_vel[1], self.motion_sig)
		self.vel = np.asarray([vx, vy])
		# TODO : this is vel in global local frame!!
		
		motion = self.vel*self.dt
		self.prev_measure = self.new_measure + 0.0
		
		# MOTION UPDATE, with uncertainty
		x , self.sig = kf_predict(self.target_pose[0], self.sig, motion[0], self.motion_sig)
		y , self.sig = kf_predict(self.target_pose[1], self.sig, motion[1], self.motion_sig)
		self.target_pose = np.asarray([x, y])
		# ~ print('Predict: [{}, {}]'.format(mu, sig))
	
# COMBINE GAUSSIANS	
def kf_update(mean1, var1, mean2, var2):
	# ~ print(var2, mean1)
	# ~ print(var1, mean2)
	new_mean = (var2*mean1 + var1*mean2)/(var2+var1) # weighted sum of means
	new_var = 1/(1/var2 + 1/var1)						# new variance
	
	return [new_mean, new_var]


# the motion update/predict function
def kf_predict(pose, pose_var, motion, motion_var):
	''' This function takes in two means and two squared variance terms,
		and returns updated gaussian parameters, after motion.'''
	# Calculate the new parameters
	new_mean = pose + motion
	new_var = pose_var + motion_var		
	
	return [new_mean, new_var]
		
if __name__ == '__main__':
	try:
		target_tracker()
	except rospy.ROSInterruptException:
		pass


