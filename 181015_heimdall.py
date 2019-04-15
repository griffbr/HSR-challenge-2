#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as pickle; import IPython; import sys; import time; from copy import deepcopy; import time
import cv2; import numpy as np; import glob; import os; import math; import scipy
from skimage.measure import label
import matplotlib.pyplot as plt
import sys
from osvos import *
from data_subscriber import image_subscriber, state_subscriber, odometry_subscriber
from cyclops import *
from misc_functions import *
import hsrb_interface;
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState, PointCloud2
from nav_msgs.msg import Odometry

from gm import *

## Functions. #########################################################################

def grab_target(whole_body, base, gripper, OSVOS, img_sub, name, 
	mask_center_init, idx, dirs):
	w_max_iv = 350; w_min_wr = 235
	s_des_iv = [240, 330]; s_des_ps3_x_grasp = [240, 220];
	s_des_ps3_tri = [240, 180]
	Je_pinv = np.array([[0, -0.001],[0.001, 0]])
	Je_pinv_tri = np.array([[0, -0.001],[0.0002, 0]])
	Je_pinv_lat = np.array([[0, 0],[0.001, 0]])
	error_high = 50
	object_grasped = False
	if mask_center_init[1] < w_max_iv:
		center_target_on_pose(whole_body, base, gripper, OSVOS, img_sub, 
					'./data/initial_view.pk', s_des_iv, Je_pinv*1.2,
					error_max = error_high)
		while not object_grasped:
			center_target_on_pose(whole_body, base, gripper, OSVOS, img_sub, 
					'./data/ps3_x_grasp.pk', s_des_ps3_x_grasp, Je_pinv*0.3)
			if mask_center_init[1] > w_min_wr:
				try: rotate_wrist(whole_body, OSVOS, img_sub, dirs)
				except: print('Could not rotate wrist.')
			object_grasped = grab_and_check(whole_body, gripper, OSVOS, img_sub, dirs, idx)
	else:
		center_target_on_pose(whole_body, base, gripper, OSVOS, img_sub, 
							'./data/initial_view.pk', s_des_iv, Je_pinv_lat*1.2,
							error_max = error_high)
		while not object_grasped:
			center_target_on_pose(whole_body, base, gripper, OSVOS, img_sub, 
					'./data/ps3_tri_tilt_low.pk', s_des_ps3_tri, Je_pinv_tri)
			object_grasped = grab_and_check(whole_body, gripper, OSVOS, img_sub, dirs, idx)

def center_target_on_pose(whole_body, base, gripper, OSVOS, img_sub, pose_file, 
							s_des, Je_pinv, pose_filter = [], error_max = 30):
	print('center_target_on_pose')
	load_pose(whole_body, pose_file, pose_filter)
	error_sum = 100
	error_logic = abs(Je_pinv).sum(axis=0) > 0
	while error_sum > error_max:
		mask, n_mask_pxls, grasp_img = get_mask(img_sub, OSVOS) # TODO remove grasp_img return
		write_seg_image(grasp_img, mask, './img/center.png') # TODO remove
		if n_mask_pxls < 200:
			move_joint_amount(whole_body, 'arm_lift_joint', 0.01)
			base.go_rel(-0.02,0,0)
			mask = []
		else:
			mask_center = find_mask_centroid(mask)
			error = s_des - mask_center
			delta_state = np.matmul(Je_pinv, error)
			base.go_rel(delta_state[0], delta_state[1], 0)
			error_sum = np.sum(abs(error*error_logic)) 
			error_max *= 1.05

def grab_and_check(whole_body, gripper, OSVOS, img_sub, dirs, idx):
	print('grab_and_check')
	#gripper.apply_force(1.0)
	#gripper.command(-0.5)
	object_grasped = False
	try:
		print('smart grasp')
		init_grasp = smart_grasp(whole_body, gripper)
		if init_grasp > -0.8 or idx == 3:
			print('load grasp check')
			try:
				load_pose(whole_body, './data/gc2.pk', ['hand_motor_joint'])
			except:
				print('except, raising torso then grasp checking')
				move_joint_amount(whole_body, 'arm_lift_joint', 0.1)
				load_pose(whole_body, './data/gc2.pk', ['hand_motor_joint'])
			#move_joint_amount(whole_body, 'wrist_flex_joint', 1.9)
			mask, n_mask_pxls, grasp_img = get_mask(img_sub, OSVOS)
			write_seg_image(grasp_img, mask, './img/g_check.png') # TODO remove
			print('Number of pixels for primary target is ' + str(n_mask_pxls))
			if n_mask_pxls > 3000:
				combined_mask = check_all_objects(OSVOS, dirs, grasp_img)
				write_seg_image(grasp_img, combined_mask, './img/g_check_all.png') # TODO remove
				combined_mask, n_comb_pxls = largest_region_only(combined_mask)
				print('Number of combined pixels is ' + str(n_comb_pxls))
				#if mask_count(combined_mask) < n_mask_pxls * 1.2:
				if n_comb_pxls < n_mask_pxls * 1.2:
					print('Object grasped')
					object_grasped = True
				else: 
					print('Other object detected in grasp')
					load_pose(whole_body, './data/drop_pose.pk', ['hand_motor_joint'])
					gripper.command(1.0)
		else:
			print('Grasp missed')
			gripper.command(1.0)
			whole_body.move_to_joint_positions({'wrist_roll_joint':0.0})
	except:
		print('grab_and_check exception')
		gripper.command(1.0)
		whole_body.move_to_joint_positions({'wrist_roll_joint':0.0})
	return object_grasped

def rotate_wrist(whole_body, OSVOS, img_sub, dirs):
	grasp_img = deepcopy(img_sub._input_image)
	print('rotate wrist')
	combined_mask = check_all_objects(OSVOS, dirs, grasp_img)
	write_seg_image(grasp_img, combined_mask, './img/rotate.png') # TODO remove
	grasp_angle = np.radians(select_grasp_angle(combined_mask,'./data/g_msk/'))
	print ('Grasp angle is %5.3f.' % grasp_angle)
	wrist_angle = whole_body.joint_positions['wrist_roll_joint']
	new_wrist_angle = (wrist_angle - grasp_angle)
	new_wrist_angle = grasp_angle_to_pm90(new_wrist_angle, angle_mod=1.5708)
	print ('New wrist angle is %5.3f.' % new_wrist_angle)
	whole_body.move_to_joint_positions({'wrist_roll_joint':new_wrist_angle})

def check_all_objects(OSVOS, dirs, img):
	init_dir = deepcopy(OSVOS.checkpoint_file)
	init_mask = OSVOS.segment_image(img)
	n_object = len(dirs)
	masks = [[] for _ in range(n_object)]
	for i, dir_i in enumerate(dirs):
		if dir_i == init_dir:
			masks[i] = init_mask
		else:
			OSVOS.change_model(dir_i)
			masks[i] = OSVOS.segment_image(img)
	OSVOS.change_model(init_dir)
	return combine_masks(masks)

########################################################################
def place_target(whole_body, base, gripper, img_sub, cab_pos, home_pose):
	placed = False
	bn = False
#	go_to_amcl_pose(base, home_pose)
#	go_to_amcl_pose(base, home_pose)
	go_to_map_home(base)
	go_to_map_home(base)
	if cab_pos == 'tl': 
		load_pose(whole_body, './data/tl2.pk',['hand_motor_joint'])
		action_sequence = [[0,0.52,-0.7854], [0.22,0.22,0]]
		back_sequence = [[-0.22,-0.22,0]]
	elif cab_pos == 'tr':
		load_pose(whole_body, './data/tr2.pk',['hand_motor_joint'])
		action_sequence = [[0,-0.58,0.7854], [0.30,-0.30,0]]
		back_sequence = [[-0.30,0.30,0]]
	elif cab_pos == 'tll':
		load_pose(whole_body, './data/tl2.pk',['hand_motor_joint'])
		action_sequence = [[0,0.77,-0.7854],[0.22,0.22,0]]
		back_sequence = [[-0.22,-0.22,0]]
	elif cab_pos == 'trb':
		load_pose(whole_body, './data/trb2.pk',['hand_motor_joint'])
		action_sequence = [[0,-0.58,0],[0.32,0,0]]
		back_sequence = [[-0.32,0,0]]
		bn = True
	while not placed:
		placed = dead_reckon_and_back(whole_body, base, gripper, action_sequence, back_sequence, bn)
#		go_to_amcl_pose(base, home_pose)
		go_to_map_home(base)

def dead_reckon_and_back(whole_body, base, gripper, sequence, back_sequence, bn=False):
	try:	
		for i, action in enumerate(sequence):
			print (action)
			base.go_rel(action[0],action[1],action[2])
			amcl_pose = get_amcl_pose()
			print('amcl position is ' + str(amcl_pose))
		if bn:
			load_pose(whole_body, './data/mic_d2.pk')
			#whole_body.move_to_joint_positions({'wrist_roll_joint': -1.5708})
		else:
			gripper.command(1.0)
		placed = True
	except:
		print('Dead reckon exception')
		placed = False
	for i, action in enumerate(back_sequence):
		print (action)
		base.go_rel(action[0],action[1],action[2])
	return placed

# TODO

## Main. ##############################################################################
def main(OSVOS, grasp_img_sub, robot_state_sub, base_odometry_sub, base, whole_body):
	gripper = robot.get('gripper')
	names = ['r','y','b','bn']
	r_dir = './data/r.ckpt-10001'
	y_dir = './data/y.ckpt-10001'
	b_dir = './data/b.ckpt-10001'
	bn_dir = './data/bn.ckpt-10001'
	dirs = [r_dir, y_dir, b_dir, bn_dir]
	cab_pos = ['tll','tr','tl','trb']
	ol = [[0,0,255],[0,255,255],[255,0,0],[0,255,0]] # TODO remove
	min_pxls = [500,500,500,2000]
	n_targets = len(names)

	home_amcl_pose = []
	if 1:
		print('\n\nwarning!!!! skipping initialization!!\n\n')
	else:
		init_amcl_fast(base, 0.5)
	#home_amcl_pose = get_amcl_pose()

	print ('\n\nRobot moves next, make sure you are ready!\n\n')
	IPython.embed()

	loop_count = 0
	while True:
		load_pose(whole_body, './data/initial_view.pk')
		grasp_img = deepcopy(grasp_img_sub._input_image)
		masks = [[] for i in range(n_targets)]
		n_mask_pxls = np.zeros(n_targets)
		mask_centers = np.zeros(shape=(n_targets,2))
		for i, name in enumerate(names):
			OSVOS.change_model(dirs[i])
			masks[i] = OSVOS.segment_image(grasp_img)
			masks[i], n_mask_pxls[i] = largest_region_only(masks[i])
			mask_centers[i] = find_mask_centroid(masks[i])
			write_seg_image(grasp_img, masks[i], './img/'+names[i]+'.png',ol[i]) # TODO remove
		bn_bias = [False, False, False, True]
		pxl_check = n_mask_pxls > min_pxls
		lat_place = [a or b for a,b in zip(mask_centers[:,0]<190,  
														mask_centers[:,0]>430)]
		target_order = np.argsort(mask_centers[:,1] - 10000*pxl_check + 
			2*mask_centers[:,1]*lat_place + -3*mask_centers[:,1]*bn_bias)
		idx = target_order[0]
		print ('Number of pixels are ' + str(n_mask_pxls))
		print ('Target index is ' + str(idx))
		if n_mask_pxls[idx]>min_pxls[idx]:
			OSVOS.change_model(dirs[idx])
			grab_target(whole_body, base, gripper, OSVOS, grasp_img_sub, 
								names[idx], mask_centers[idx], idx, dirs)
			place_target(whole_body, base, gripper, grasp_img_sub, 
											cab_pos[idx], home_amcl_pose)
		'''
		for i, idx in enumerate(target_order):
			if n_mask_pxls[idx]>min_pxls[idx]:
				OSVOS.change_model(dirs[idx])
				grab_target(whole_body, base, gripper, OSVOS, grasp_img_sub, 
									names[idx], mask_centers[idx], idx, dirs)
				place_target(whole_body, base, gripper, grasp_img_sub, 
												cab_pos[idx], home_amcl_pose)
		print ('Finished grabbing all blocks!')
		'''
		loop_count += 1
		if loop_count > 10:
			base.go_rel(0,0.05,0)
	IPython.embed()

if __name__ == '__main__':
	with hsrb_interface.Robot() as robot:
		base = robot.try_get('omni_base')
		whole_body = robot.get('whole_body')
		grasp_img_sub = image_subscriber('/hsrb/hand_camera/image_raw', True)
		robot_state_sub = state_subscriber('/hsrb/joint_states')
		base_odometry_sub = odometry_subscriber('/hsrb/odom')
		OSVOS = osvos_seg('./data/r.ckpt-10001')	
		main(OSVOS, grasp_img_sub, robot_state_sub, base_odometry_sub, base, whole_body)
