''' Script for visualizing KITTI scene flow point cloud

Author: Xingyu Liu
Date: Dec 2019
'''


import numpy as np
import mayavi.mlab as mlab
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='E:/fyp/dataset/output/000010.npz', help='Filename to visualize [default: kitti_flow/000012.npz]')
# parser.add_argument('--filename', default='E:/fyp/meteornet-master/scene_flow_kitti/kitti_flow/000012.npz', help='Filename to visualize [default: kitti_flow/000012.npz]')
FLAGS = parser.parse_args()


data = np.load(FLAGS.filename)

points1 = data['points1']
points_mid = data['points_mid']
points2 = data['points2']
points3 = data['points3']
points4 = data['points4']
flow = data['flow']
mask = data['mask']

point_size = 0.1

mlab.figure(bgcolor=(1,1,1))
# mlab.points3d(points1[:, 0], points1[:, 1], points1[:, 2], color=(1, 0, 0), scale_factor=point_size)
# mlab.points3d(points2[:, 0], points2[:, 1], points2[:, 2], color=(1, 1, 0), scale_factor=point_size)
mlab.points3d(points_mid[:, 0], points_mid[:, 1], points_mid[:, 2], color=(0, 1, 0), scale_factor=point_size)
# mlab.points3d(points3[:, 0], points3[:, 1], points3[:, 2], color=(0, 1, 0), scale_factor=point_size)
# mlab.points3d(points4[:, 0], points4[:, 1], points4[:, 2], color=(0, 0, 1), scale_factor=point_size)
flowed = points1 + flow
warped_points1_xyz = points1 + flow * 0.5
mlab.points3d(warped_points1_xyz[:, 0], warped_points1_xyz[:, 1], warped_points1_xyz[:, 2], color=(1, 0, 0), scale_factor=point_size)
# mlab.points3d(flowed[:, 0], flowed[:, 1], flowed[:, 2], color=(0, 0, 0), scale_factor=point_size)

mlab.view()
input()


