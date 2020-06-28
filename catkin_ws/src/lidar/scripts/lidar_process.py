#!/usr/bin/env python

# processes sensor data
# Author = David Isele


import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#from DeepRL.dqn.SumoCarMDP import *
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, PoseArray, Polygon, Point, Pose2D, Pose
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2
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

VIS_CLUSTER = 0
VIS_HULL = 0

def default_marker(x=0,y=0,z=0):
	m = Marker()
	m.header.frame_id = "laser"
	m.type = 4 # 4 line strip, 6 cube list # car.type
	m.scale.x = .10
	m.scale.y = .10
	m.scale.z = .10
	m.pose.position.x=x
	m.pose.position.y=y
	m.pose.position.z=z
	m.pose.orientation.x=0
	m.pose.orientation.y=0
	m.pose.orientation.z=0
	m.pose.orientation.w=1
	m.color.r = 0.7
	m.color.g = 0.7
	m.color.a = 0.7

	return m

class lidar_processing():
	def __init__(self):

		## SETUP ROS NODES
		rospy.init_node('lidar_processing', anonymous=False)
		rospy.loginfo("To stop vecs_to_gonogo CTRL + C")  
		rospy.on_shutdown(self.shutdown)
		self.r = rospy.Rate(10) # 10hz

		self.coms = 0.0
		self.vel = 0.0
		
		new_pose = np.asarray([1,0])
		target_pose = np.asarray([1,0])
		
		#SUBSCRIBER
		#rospy.Subscriber('/converted_pc', PointCloud2, self.parse_command)
		self.laser_projector = LaserProjection()
		self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.on_scan)
		# target pose from filter
		
		# PUBLISHER
		# self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		# ~ self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)
		self.pub_hulls = rospy.Publisher('/convex_hulls', MarkerArray, queue_size=1)
		# ~ self.pub_new_target = rospy.Publisher('/new_target', Point, queue_size=1)
		self.pub_new_target = rospy.Publisher('/new_target', Marker, queue_size=1)		
		
		rospy.sleep(0.01) 
		
		rospy.loginfo("Waiting for topics...")
		try:
			rospy.wait_for_message("/scan", LaserScan, timeout=10.0)
		except rospy.ROSException as e:
			rospy.logerr("Timeout while waiting for topics!")
			raise e
		print("Node started")
		
		while not rospy.is_shutdown():
			self.r.sleep()
			#self.xyz_generator
			
			# GET CLUSTERS
			points, labels, n_clusters = self.get_clusters()
			
			# GET HULLS and their COMs
			hulls = []
			coms = []
			for n in range(n_clusters):
				hull = ConvexHull(points[labels==n,:])
				# ~ print("mean",np.mean(points[labels==n,:], 0) )
				coms.append(np.mean(points[labels==n,:], 0) )
				# hull.vertices, hull.area
				# print(n, 'area', hull.area, np.std(points[labels==n,:],0))
				hulls.append(hull)
			
			# GET TARGET POSE
			dists = np.sum((np.vstack(coms)-new_pose)**2,1)
			self.new_id = np.argmin(dists)
			
			
			# PUBLISH HULLS
			marker = self.get_hull_markers(hulls, coms)
			self.pub_hulls.publish(marker)
			
			# PUBLISH NEW TARGET (for reset condition)
			nt = coms[self.new_id]
			# ~ matched = coms[self.cluster_id]
			# ~ print("new target", nt, nt[0], nt[1])
			# ~ self.pub_new_target.publish(Pose(Point(x=nt[0], y=nt[1],z=0) ) ) 
			m = default_marker(x=nt[0], y=nt[1],z=0)
			m.type = 2
			m.color.r = 0
			m.color.a = 1.0
			m.scale.x = .30
			m.scale.y = .30
			m.scale.z = .30
			self.pub_new_target.publish(m)  

			
			
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
			
			# ~ self.pub.publish(self.data) # numpy in ros only handles 1d arrays


	def shutdown(self):
		rospy.loginfo("Stop lidar_processing")
		# rospy.sleep(1)
		return 0

	def get_clusters(self):
		# ~ print(data.data)
		#for p in gen:
		#	print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])
		self.xy = deepcopy(self.xyz[:,:2])
		db = DBSCAN(eps=0.15, min_samples=3).fit(self.xy)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise = list(labels).count(-1)
		self.labels = labels
		
		if VIS_CLUSTER:
			fig = plt.figure(1)
			plt.clf()

			for c in range(n_clusters):
				col = np.random.uniform(0,1,(3))
				# ~ print(np.shape(self.xy[labels==c,0]))
				plt.plot(self.xy[self.labels==c,0], self.xy[self.labels==c,1], '.', color = col)
				# ~ plt.plot(self.vel_hist, '.r')
			plt.plot(self.xy[self.labels==-1,0], self.xy[self.labels==-1,1], '.', color = [0,0,0])
			plt.xlabel("Time")
			plt.ylabel("Speed (m/s)")
			plt.show(block=False)
			plt.pause(0.0000000001)
			
		# ~ print("clusters:",n_clusters)
		return self.xy, self.labels, n_clusters
	
	def on_scan(self, scan):
		#print scan
		#rospy.loginfo("Got scan, projecting")
		cloud = self.laser_projector.projectLaser(scan)
		#print cloud
		#rospy.loginfo("Printed cloud")
		gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
		self.xyz_generator = gen
		self.xyz = np.asarray([[p[0],p[1],p[2]] for p in gen])
		# ~ print(np.shape(self.xyz))
		#for p in gen:
		#	print " x : %f  y: %f  z: %f" %(p[0],p[1],p[2])

	def get_hull_markers(self, hulls, coms):
	
		marker = MarkerArray()
		
		ind = 0
		for h in range(len(hulls)):
			m = default_marker()
			m.id = ind #car.id -- all car.ids are 0
			m.lifetime = rospy.Duration(0.1)
			# ~ m.header = car.header
			# m.color.g = 0.72
			# m.color.a = (car.id+0.0)/t
			# m.scale.x = 1.0 + 5*(car.id+0.0)/t
			# m.scale.y = 1.0 + 5*(car.id+0.0)/t
			# ~ print("hull",hull)
			
			m.points.append(Point(x=coms[h][0], y=coms[h][1], z=0))
			for v in hulls[h].vertices:
				m.points.append(Point(x=hulls[h].points[v,0], y=hulls[h].points[v,1], z=0)) 

			marker.markers.append(m)
			ind +=1

		return marker
		
if __name__ == '__main__':
	try:
		lidar_processing()
	except rospy.ROSInterruptException:
		pass


