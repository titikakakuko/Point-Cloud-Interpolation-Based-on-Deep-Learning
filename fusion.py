import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os
# from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from dcp import *


def trans_pointcloud(pointcloud):
    # [:, :3]
    # pointcloud = self.data[item][:self.num_points]
    # if self.gaussian_noise:
    #     pointcloud = jitter_pointcloud(pointcloud)
    # if self.partition != 'train':
    #     np.random.seed(item)
    factor = 4
    anglex = np.random.uniform() * np.pi / factor
    angley = np.random.uniform() * np.pi / factor
    anglez = np.random.uniform() * np.pi / factor

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)
    R_ba = R_ab.T
    translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                np.random.uniform(-0.5, 0.5)])
    translation_ba = -R_ba.dot(translation_ab)

    pointcloud1 = pointcloud.T

    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

    euler_ab = np.asarray([anglez, angley, anglex])
    euler_ba = -euler_ab[::-1]

    pointcloud1 = np.random.permutation(pointcloud1.T).T
    pointcloud2 = np.random.permutation(pointcloud2.T).T

    return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
            translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
            euler_ab.astype('float32'), euler_ba.astype('float32')


def get_trans_matrix(net,src,target):
    src = next(iter(src))
    target = next(iter(target))

    rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

    # ## save rotation and translation
    # rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
    # translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
    # ##
    # rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
    # translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())

    transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

    transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

    ###########################

    batch_size = src.size(0)
    identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
    translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                    + translation_ba_pred) ** 2, dim=[0, 1])
    cycle_loss = rotation_loss + translation_loss

    transformed_src = torch.tensor(transformed_src, dtype=torch.float32).cuda()
    target = torch.tensor(target, dtype=torch.float32).cuda()
    transformed_target = torch.tensor(transformed_target, dtype=torch.float32).cuda()
    src = torch.tensor(src, dtype=torch.float32).cuda()


    mse_ab = torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item()
    mae_ab = torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() 

    mse_ba = torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item()
    mae_ba = torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item()

    # rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    # translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    # rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    # translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    print('A--------->B')
    print('Loss: %f, Cycle Loss: %f, mse_ab: %f, mae_ab: %f, mse_ba: %f, mae_ba: %f'
                  % (translation_loss, cycle_loss, mse_ab, mae_ab, mse_ba, mae_ba))

    return mse_ab, mae_ab, mse_ba, mae_ba, rotation_ab_pred, translation_ab_pred, \
           rotation_ba_pred, translation_ba_pred, rotation_loss, translation_loss, cycle_loss

