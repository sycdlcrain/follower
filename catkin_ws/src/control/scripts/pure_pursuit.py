import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import minimize
import time

width = 1.2
height = 0.6

########### HERMITE SPLINE 2D ##############

def get_spline(x0, x1, y0, y1, theta0, theta1, steps=100):
	t = np.linspace(0, 1, steps) # np.asarray([1, 5, 10]) #

	# X = T*b
	# x  =   a*t**3 + b*t**2 + c*t + d
	# x' = 3*a*t**2 + 2*b*t  + c
	# x0 = 0  
	# y0 = 0
	# theta0 = 0
	dx0 = np.cos(theta0) # = c  # all change is at along x at start
	dy0 = np.sin(theta0)
	# x1 = 2  
	# y1 = 1
	# theta1 = 0+np.random.uniform(0.001, 0.01)
	dx1 = np.cos(theta1) # = c  # all change is at along x at start
	dy1 = np.sin(theta1)

	t0 = 0
	t1 = 1

	# TREAT X AND Y SEPARATE, SO THERE'S NOT A SINGULARITY

	Ax = np.asarray([[1, t0,   t0**2,   t0**3],  # x  @ 0
					[0, 1,  2*t0,    3*t0**2],  # x' @ 0
					[1, t1,   t1**2,   t1**3],  # x  @ 1
					[0, 1,  2*t1,    3*t1**2]]) # x' @ 1

	X = np.asarray([x0, dx0, x1, dx1]).transpose()
	bx = np.linalg.solve(Ax, X)

	Ay = np.asarray([[1, t0,   t0**2,   t0**3],  # x  @ 0
					[0, 1,  2*t0,    3*t0**2],  # x' @ 0
					[1, t1,   t1**2,   t1**3],  # x  @ 1
					[0, 1,  2*t1,    3*t1**2]]) # x' @ 1
	Y = np.asarray([y0, dy0, y1, dy1]).transpose()
	by = np.linalg.solve(Ay, Y)


	x = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(),bx)
	y = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(),by)

	return x, y

def spline_length(x,y):
	xdiff = x[1:]-x[0:-1]
	ydiff = y[1:]-y[0:-1]
	dists = np.sqrt(xdiff**2 + ydiff**2)
	cumsum = np.cumsum(dists)
	length = np.sum(dists)
	return length, cumsum

##########################################

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

def update_pose(state, action, dt=0.1):

	# maxv = 1.0 ## max velocity
	# minv = 0.0 ## min velocity, don't let it go backwards on the merge
	# max_axle = np.pi/4 
	dist_rear_axle = width/2
	dist_front_axle = width/2

	acceleration_index = 0
	steering_index = 1

	# Kong et al. Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design. IV 2015 
	# [x, y, theta, v, beta(com)]
	# BOUND ACTIONS
	max_accel = 1.0 
	max_angle = 1.4
	if np.abs(action[0])>max_accel:
		action[0] = np.sign(action[0])*max_accel
	if np.abs(action[1])>max_angle:
		action[1] = np.sign(action[1])*max_angle

	# x and y 
	state[0] += state[3]*np.cos(state[2]+state[4]) * dt  # x
	state[1] += state[3]*np.sin(state[2]+state[4]) * dt  # y

	# theta
	state[2] += state[3]/dist_rear_axle * np.sin(state[4]) * dt                  

	# v
	k = 0.2
	state[3] += (action[acceleration_index]-k*state[3]) * dt   # make velocity has damping
	# if np.abs(state[3]) > maxv:
	# 	state[3] = np.sign(state[3]) * maxv
	# if state[3] < minv:
	# 	state[3] = minv

	# if np.abs(action[steering_index]) > max_axle :
	# 	print("excess")
	# 	action[steering_index] = np.sign(action[steering_index]) * max_axle

	# state[4] = np.arctan(dist_rear_axle/width) * np.tan(action[steering_index]) * dt # this is a position control
	state[4] = np.arctan( (dist_rear_axle/(dist_rear_axle+dist_front_axle)) * np.tan(action[steering_index]) )  # this is a position control
	return state

def get_trajectory(state, action, timesteps):
	# produce a trajectory prediction given the current action (accel, steer)
	traj = []
	traj.append(state+0.0)
	for i in range(timesteps):
		state = update_pose(state, action) + 0.0
		traj.append(state+0.0)
	return np.asarray(traj)

def my_func(action, args): # MPC FUNCTION TO OPTIMIZE
	st, goal, timesteps = args
	# print("state", st, goal)
	trajectory = get_trajectory(st+0.0, action, timesteps)
	# cost = np.sum((trajectory[-1,:]-goal)**2)
	cost = np.sum((trajectory[1:,:2]-goal[:,0:2])**2)
	# print("cost", cost, "action", action)
	# print cost,
	return cost

def find_goal(person):
	# offset_forward = -0.2
	# offset_side = -0.3
	offset_dist = .8
	offset_angle = 7.0/8.0*np.pi
	
	# ADD PREDICTION, smooth change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	goal = person+0.0
	theta = person[2]+offset_angle
	# print(theta*180.0/2*(np.pi))
	goal[0] += offset_dist*np.cos(theta)
	goal[1] += offset_dist*np.sin(theta)
	return np.hstack((goal,[0]))

def update_person(person, dt):
	person[0] += person[3]*dt*np.cos(person[2])
	person[1] += person[3]*dt*np.sin(person[2])
	return person

def bound_action(action):
	action[0] = np.min([action[0],1.0])
	action[0] = np.max([action[0],-1.0])
	action[1] = np.min([action[1],1.4])
	action[1] = np.max([action[1],-1.4])
	return action

def simple_control(ego_state, target, spline):

	print("ego", np.shape(ego_state))
	print("target", np.shape(target))
	print("spline", np.shape(spline))

	# TUNABLE PARAMS
	v0 = 1.0 #speed limit
	d0 = 0.4 #minimal gap distance
	T = 0.5 #minimal time gap
	c1 = 1.0 #maximal acceleration
	c2 = 0.2 #comfortable deceleration


	# FIND ANGLE INFO
	converge = 2.0 # look ahead
	dists = np.sqrt(np.sum((spline[1:]-spline[0:-1,:])**2,1))
	sums = np.cumsum(dists)
	ind = np.argmax(sums>converge) # argmax returns the first entry 
	vec = spline[ind,:]-spline[0,:]
	desired_angle = np.arctan2(vec[1],vec[0])

	# INTELLIGENT DRIVER MODEL
	# [x, y, theta, v, com_rot_angle] * agents
	this_car = ego_state
	v = this_car[3]
	dv = this_car[3]-target[3] # order? we subtract f, so if front car is slower we should subtract more 

	b = 4
	f = d0 + v*T + (v*dv)/(2*np.sqrt(c1*c2))
	# d = (target[0]-width/2) - (this_car[0]+width/2) 
	d = (sums[-1]-width/2) - (0+width/2) 
	accel = c1*(1-(v/v0)**b - (f/d)**2)
	# v_control = ego_obs[4] + accel*dt
	# print("accel", accel, f, d )


	# y_error = (goal[1]-ego_obs[1]) # perpendicular
	# desired_angle = np.arctan2(y_error, x_converge)

	const = 7.0
	pcontrol_angle = (desired_angle - ego_state[2])*const
	
	# print(" angle", desired_angle, ego_state[2], pcontrol_angle)
	max_angle = 1.4
	if pcontrol_angle>np.abs(max_angle):
		action = [-accel, -np.sign(pcontrol_angle)*max_angle]
	else:
		action = [accel, pcontrol_angle]
	# print("action", action)


	return action

def globalToLocal(center, theta, p):
	delta = np.zeros(2)
	delta[0] = p[0] - center[0]
	delta[1] = p[1] - center[1]
	return rotate(delta, -theta)

def localToGlobal(center, theta, p): 
	out = rotate(p, theta)
	out[0] += center[0]
	out[1] += center[1]
	return out

def rotate(pt, theta):
	s = np.sin(theta)
	c = np.cos(theta)

	out = np.zeros(2)
	out[0] = pt[0] * c - pt[1] * s
	out[1] = pt[0] * s + pt[1] * c
	# print("out",out)
	return out

def pure_pursuit(ego, target, spline):
	# ego is [x, y, theta, v, beta(com)]
	# target [x, y, theta, v, 0]
	# path is num_pts x [x,y]

	# # Closest point to path
	# dist = np.sum((ego[:2]-spline)**2, 1)
	# close_id = np.argmin(dist)

	look_ahead = 0.3	
	length, cumsum = spline_length(spline[:,0],spline[:,1])
	try:
		goal_id = np.argmax(cumsum>look_ahead) # max returns first 
	except:
		print("spline_length", len(length), length[-1])
		goal_id = len(length)-1

	# CURVATURE CALCULATION
	# x = forward, y is lateral
	#				y
	# |---------|	|
	# |			|   ------x
	# |---------|	
	#				
	#				

	local_goal = globalToLocal(ego[:2], ego[2], spline[goal_id,:])

	straight_dist_sq = np.sum(local_goal**2)
	# print("::",straight_dist_sq, local_goal)

	version = 1
	if version==0:
		print("This is currently broken")
		r = straight_dist_sq/(2*local_goal[0])	
		curvature = 1.0/r
		# s = wheel_base
		# a = angle
		# n = steering ratio
		# 1/r= sin(a/n)/s ## NOPE!
		# r = s / (sqrt(2 - 2 * cos(2*a/n)) # NOPE 
		#Reference https://www.physicsforums.com/threads/steering-wheel-angle-radius-of-curvature.59881/

		wheel_base = .8 #height -0.2
		steer_ratio = 1.0
		# angle = np.arcsin(curvature*wheel_base)/steer_ratio
		# angle = np.arcsin(((wheel_base/r)**2 - 2.0)/2.0)*steer_ratio / 2.0 
		try:
			angle = np.arcsin(wheel_base/r) # I get this when I calc both wheel on the circle, but i need to use arctan2 to get the sin right
			x = np.sqrt(r**2 - wheel_base**2) 
			angle = np.arctan2(wheel_base, x) # TODO: still prob 1111111111111111111111111111111111111111111111111111111111111
		except:
			angle = 0
		# print("::",r, curvature, angle)
		action = [0.1, angle]
	elif version==1:
		# # CALCULATE PURE PURSUIT
		# # local coords
		x = local_goal[0]
		y = local_goal[1]
		l2 = (x**2+y**2)
		twox = 2*y
		r = (l2+0.0)/twox
		# print("x",x, "y", y)

		# # if x is negative, we need to go in the other direction
		s = np.sign(y)
		# print("l2", l2,"2x",twox, "r", r, "signed extent", s*width/2)

		# # if y is neg, the same arc will get you there as if it were positive
		angle = np.arctan2(s*width/2, np.abs(r))
		# print("angle", angle)

		# speed is a function of target speed, target distance, and curvature
		goal_dist = np.sqrt(np.sum((spline[-1,:2] - ego[:2])**2))
		q = 1.0

		# action = [q/goal_dist, angle]
		# action = [0.2, angle]


 
	return angle


def get_accel(state, goal, goal_speed, max_speed, dt, max_accel=1.0):
	distance_to_goal = np.sqrt(np.sum((state[0:2]-goal[0:2])**2))

	# follower needs to be behind
	eps = 0.3
	goal_speed = np.maximum(goal_speed-eps,0.0)

	if goal_speed>max_speed:
		tv = max_speed
	else:
		# linear 

		c = 5.0
		d = np.minimum(distance_to_goal,c)
		ratio = d/c
		# print("dist", d, ratio)
		tv = (1-ratio)*goal_speed + ratio*max_speed
	

	v_error = tv - state[3]
	accel = np.minimum(max_accel, v_error/dt)
	# print("1--", max_accel, v_error/dt, accel)
	accel = np.maximum(-max_accel, accel)
	# print("2--", -max_accel, accel, accel)
	# print("tv", tv, "err", v_error)
	# print(ratio, "v", state[3], goal_speed, "a", accel)
	return accel
##################################################################################


print("Person Following targeting closeness to spline")

dt = 0.1

person = np.asarray([8.0, 5.0, 0.0, 0.5]) # [x,y, theta(heading)]

state0 = np.asarray([20.0, 12.0, 0.0, 0.5, 0.0]) # [x,y, theta(heading), v, beta(com)]
# state0 = np.asarray([0.0, 10.0, 0.0, 0.5, 0.0]) # [x,y, theta(heading), v, beta(com)]
state = state0 + 0.0
action0 = np.asarray([0.5, 0.0]) # make previous action ?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
action = np.asarray([0.0, 0.0])
max_speed = 1.0

while 1:          
	start = time.time()

	state = update_pose(state, action, dt)
	person = update_person(person, dt)
	goal = find_goal(person) # offset position

	# BOUND ACTION
	# max 1.0, max abs(1.4)
	# action = np.asarray([0.8, 1.4]) # accel, steering (position) # at angle=1.5 the opt gets shitty
	distance_to_goal = np.sqrt(np.sum((state[0:2]-goal[0:2])**2)) # FIX TO BE NON-EUCLIDEAN 
	target_speed = np.mean([person[3], max_speed])
	timesteps = 50 # int(np.min( [np.ceil(distance_to_goal/target_speed/dt), 50] ) )
	# print(state,goal,timesteps)

	x, y = get_spline(state[0], goal[0], state[1], goal[1], state[2], goal[2], steps=timesteps) # not necessarily feasible
	traj = np.vstack([x,y]).transpose()

	# dists = np.sqrt(np.sum((traj[1:]-traj[0:-1,:])**2,1))
	# print(np.shape(dists))
	# action = [0.1,0.0]
	# action = simple_control(state, goal, traj)
	angle = pure_pursuit(state, goal, traj)
	accel = get_accel(state, goal, person[3], max_speed, dt)
	action = [accel, angle]

	# res = minimize(my_func, action0, args=[state, traj, timesteps], method='BFGS', tol=1e-3) 
	# print(":::::selected action", res.x)
	# action = res.x # bound_action(res.x)


	# VISUALIZE 
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

