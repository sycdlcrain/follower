#!/usr/bin/env python

# create occupancy map, publish trajectory
# Author = David Isele

# Huck money: 140 + 175/4

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
from nav_msgs.msg import OccupancyGrid

import matplotlib.pyplot as plt
#import tensorflow as tf

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.ndimage import gaussian_filter

from copy import deepcopy
from bresenham import bresenham # cells = np.vstack(list(bresenham(96, 280, 95, 275)))

VIS_CLUSTER = 0
VIS_MAP = 0

def default_marker(x=0,y=0,z=0):
	m = Marker()
	m.header.frame_id = "laser"
	m.type = 4 # 4 line strip, 6 cube list # car.type
	m.pose.position.x=x
	m.pose.position.y=y
	m.pose.position.z=z
	m.scale.x = .10
	m.scale.y = .10
	m.scale.z = .10
	m.pose.orientation.x=0
	m.pose.orientation.y=0
	m.pose.orientation.z=0
	m.pose.orientation.w=1
	m.color.r = 0.0
	m.color.g = 0.3
	m.color.b = 0.7
	m.color.a = 0.7

	return m


		
def raycast(grid, center, wp):		
	
	for w in range(len(wp)):
		# ~ print(w, len(wp))	
		# ~ w = 28
		cells = np.vstack(list(bresenham(center[0], center[1], wp[w,0], wp[w,1])))
		# ~ print(grid[cells[:,0],cells[:,1]])
		a = grid[cells[:,0],cells[:,1]] # ray to wall
		# ~ print(a)
		tmp  = a==1 # find obstacles in path
		# ~ print(tmp)
		cs = np.cumsum(tmp) # find where the obstacle starts
		ids = (1-cs)>0 # flip to find clear space
		grid[cells[ids,0],cells[ids,1]]=0.01 # not zero so dijkstra will find shortest path	
	return grid
		

class occupancy():
	def __init__(self):

		## SETUP ROS NODES
		rospy.init_node('occupancy', anonymous=False)
		rospy.loginfo("To stop occupancy CTRL + C")  
		rospy.on_shutdown(self.shutdown)
		self.r = rospy.Rate(10) # 10hz

		# SETUP MAP
		self.range = 10.0 # meters
		self.pixels = 36
		self.map = np.zeros((self.pixels,self.pixels))
		self.pixwid = self.pixels/self.range
		print("pixel width", self.pixwid)
		self.offset = np.asarray([1.0,5.0]) # control where 0,0 will map to
		# ~ x = np.linspace(0-self.offsets[0],self.range-self.offsets[0], self.pixels)
		# ~ y = np.linspace(0-self.offsets[1],self.range-self.offsets[1], self.pixels)
		# ~ xx,yy = np.meshgrid(x,y) 
		self.points, self.edges = self.get_graph()
		self.heuristic = np.ones(len(self.points))
		
		# MAP WORLD POINT TO MAP COORDINATE
		self.xmap = lambda a : np.minimum( np.maximum( 0,((a+self.offset[0])*self.pixwid).astype(int) ), self.pixels-1)
		self.ymap = lambda a : np.minimum( np.maximum( 0,((a+self.offset[1])*self.pixwid).astype(int) ), self.pixels-1)
		self.xinverse = lambda a : ((a+0.0)/self.pixwid)-self.offset[0]
		self.yinverse = lambda a : ((a+0.0)/self.pixwid)-self.offset[1]
		self.center = np.asarray( [self.xmap(0), self.ymap(0)] )
		wy = range(0,self.pixels)
		
		# FOR RAYCASTING (head of the 'ray')
		behind = np.vstack([np.zeros_like(wy), wy]).transpose()
		right = np.vstack([wy, np.zeros_like(wy)]).transpose()
		left = np.vstack([wy, (self.pixels-1)*np.ones_like(wy)]).transpose()
		front = np.vstack([(self.pixels-1)*np.ones_like(wy), wy]).transpose()
		self.wp = np.vstack([left, right, front])
		
		
		
		#SUBSCRIBER
		#rospy.Subscriber('/converted_pc', PointCloud2, self.parse_command)
		self.laser_projector = LaserProjection()
		self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.on_scan)
		# ~ target_string = "/new_target"
		target_string = "/tracked_target"
		self.target_sub = rospy.Subscriber(target_string, Marker, self.process_target)
		# Placeholder
		# ~ self.target = np.asarray( [28, 16] )
		# ~ self.start = np.ravel_multi_index(self.center, (self.pixels,self.pixels) )
		# ~ self.goal = np.ravel_multi_index(self.target, (self.pixels,self.pixels) )
		
		
		# PUBLISHER
		# self.pub = rospy.Publisher('/go_nogo', ObjectVector, queue_size=1)
		# ~ self.pub = rospy.Publisher('/net_input', Float32MultiArray, queue_size=1)
		self.pub_path = rospy.Publisher('/path_to_target', Marker, queue_size=1)
		# ~ self.pub_new_target = rospy.Publisher('/new_target', Point, queue_size=1)
		# ~ self.pub_new_target = rospy.Publisher('/new_target', Marker, queue_size=1)		
		
		rospy.sleep(0.01) 
		
		rospy.loginfo("Waiting for topics...")
		try:
			rospy.wait_for_message("/scan", LaserScan, timeout=10.0)
			rospy.wait_for_message(target_string, Marker, timeout=10.0)
		except rospy.ROSException as e:
			rospy.logerr("Timeout while waiting for topics!")
			raise e
		print("Node started")
		
		
		while not rospy.is_shutdown():
			self.r.sleep()

			self.update_map()
			# TODO: get heuristic
			self.start = np.ravel_multi_index(self.center, (self.pixels,self.pixels) )
			self.goal = np.ravel_multi_index(self.target, (self.pixels,self.pixels) )
			path = self.find_path(self.start, self.goal, heuristic=self.heuristic)
			
			x,y = np.unravel_index(path, (self.pixels, self.pixels))
			self.map[x,y] = 2
			
			if VIS_MAP:
				# ~ fig = plt.figure(1)
				# ~ plt.clf()
				# ~ #plt.plot(self.xy[:,0], self.xy[:,1], '.')
				# ~ plt.plot(self.xmap(self.xy[:,0]), self.xmap(self.xy[:,1]), '.')
				
				fig = plt.figure(2)
				plt.clf()
				plt.imshow(self.map)
				plt.show(block=False)
				plt.pause(0.0000000001)	
			
			
			path_message = self.get_path_message(x,y) 
			self.pub_path.publish(path_message)

	def shutdown(self):
		rospy.loginfo("Stop occupancy")
		# rospy.sleep(1)
		return 0

	def process_target(self, data):
		x = self.xmap(data.pose.position.x)
		y = self.ymap(data.pose.position.y)
		# ~ print("points", x,y)
		self.target = np.asarray( [x, y] )

	def on_scan(self, scan):
		
		cloud = self.laser_projector.projectLaser(scan)
		
		gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
		self.xyz_generator = gen
		self.xyz = np.asarray([[p[0],p[1],p[2]] for p in gen])
	
	def update_map(self):
		# ~ https://www.redblobgames.com/articles/visibility/
		# ~ 1. Calculate the angles where walls begin or end.
		# ~ 2. Cast a ray from the center along each angle.
		# ~ 3. Fill in the triangles generated by those rays.
		self.xy = deepcopy(self.xyz[:,:2])	
		xs = self.xmap(self.xy[:,0])
		ys = self.ymap(self.xy[:,1])
		# ~ print(self.xy[:,1])
		# ~ print(ys)
		self.map = 0.5*np.ones((self.pixels,self.pixels))
		self.map[xs,ys] = 1
		self.map[self.center[0],self.center[1]] = 0

		# TODO: raytracing
		self.map = raycast(self.map, self.center, self.wp)
		self.map = np.maximum(self.map, gaussian_filter(self.map, sigma=2))
		
	# https://raw.githubusercontent.com/pfirsich/Ray-casting-test/master/main.lua

	def get_graph(self):
		# GET GRAPH CORRESPONDING TO OCCUPANCY MAP

		# GET POINTS [time, forward, lateral]
		X = range(self.pixels)
		Y = range(self.pixels)
		Y, X = np.meshgrid(Y, X) # X,Y doesn't ravel correctly
		points = np.vstack( [X.ravel(), Y.ravel()] ).transpose()
		print("points shape", np.shape(points))

		# GET EDGES
		edges = []
		cost_edges = []
		# [t,x,y]-> [t+1,x:x+2,y-1:y+1] # x can only take big steps (x+0 x+2) early on

		for p in range(len(points)):
			
			x,y = np.unravel_index(p, (self.pixels, self.pixels)) #  self.x_num, self.y_num) )

			# # CONNECTIVITY
			if x==0:
				xs = [x, x+1]
			elif x==self.pixels-1:
				xs = [x-1, x]

			else:
				xs = [x-1,x,x+1]
				
			if y==0:
				ys = [y, y+1]
			
			elif y==self.pixels-1:
				ys = [y-1, y]
				
			else:
				ys = [y-1, y, y+1]
			
			xgrid, ygrid = np.meshgrid(xs, ys)
			xlist = xgrid.ravel()
			ylist = ygrid.ravel()
			
			arr = np.array([xlist,ylist]).astype(int)
			inds = np.ravel_multi_index(arr, (self.pixels, self.pixels) ) # order='F')

			edges.append(inds)

		return points, edges


	def find_path(self, start, goal, heuristic=0):
		
		# DIJKSTRA's ALGORITHM / A*
		node_num = len(self.points)
		nodedist = np.inf*np.ones(node_num)

		nodedist[start] = 0
		visited = np.zeros(node_num)
		came_from = np.zeros(node_num)

		big_num = 10000

		v = start
		for i in range(node_num):

			# STARTING NODE (working backwards)
			costs = nodedist + 10*big_num*visited + heuristic
			v = np.argmin(costs) # unvisited and closest
			# print('v', v, np.min(costs))
			if v==goal:
				break
			visited[v]=1 

			# FIND NEIGHBORS
			# k = 30 + 1  # 1st is always self
			# dist, ind = self.tree.query([self.points[v,:]], k=k)
			# ind = ind[0]
			# dist = dist[0]
			v_neighbor = self.edges[v]
			# dist = self.weights[v] # WE USE THE DESTINATION COST (from occupancy map) AS THE EDGE COST

			# COST: cost to current  + current to new cost 

			for p in range(np.size(v_neighbor)):
				# ~ line = np.asarray(np.hstack((self.points[v,:],self.points[v_neighbor[p],:])))

				new_dist = nodedist[v] + self.map.ravel()[v_neighbor[p]] 
				# print('...', p, new_dist)
				if new_dist<nodedist[v_neighbor[p]]:
					nodedist[v_neighbor[p]] = new_dist
					came_from[v_neighbor[p]] = v
			# steps+=1

		if v!=goal: #>max_steps:
			print("No path found")
			return []

		# print("costs",nodedist)
		self.node_dist = nodedist
		# RECOVER PATH
		self.path = []
		self.path.append(v)

		# FIND NEXT SUBGOAL
		# if self.subgoal!=len(self.points)-1: ################################### if not finished
		self.subgoal = start
		while came_from[v] != self.subgoal:
			v = int(came_from[v])
			self.path.append(v)
		self.path.append(self.subgoal)
		self.subgoal = v

		return self.path
		
	def get_path_message(self,x,y):
		X = self.xinverse(x)
		Y = self.yinverse(y)
	
		m = default_marker()
		m.lifetime = rospy.Duration(0.1)

		for i in range(len(X)):
			m.points.append(Point(x=X[i], y=Y[i], z=0))
		
		return m
		
if __name__ == '__main__':
	try:
		occupancy()
	except rospy.ROSInterruptException:
		pass

# https://raw.githubusercontent.com/pfirsich/Ray-casting-test/master/main.lua
# ~ https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/
# https://gamedev.stackexchange.com/questions/47576/more-efficient-way-to-implement-line-of-sight-on-a-2d-grid-with-ray-casting


# ~ grid = {}
# ~ grid.cellSize = 128
# ~ grid.width = math.ceil(love.graphics.getWidth() / grid.cellSize)
# ~ grid.height = math.ceil(love.graphics.getHeight() / grid.cellSize)
# ~ ray = {startX = grid.cellSize/2, startY = grid.cellSize/2,
# ~ dirX = grid.cellSize, dirY = grid.cellSize}
