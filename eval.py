
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mayavi.mlab as mlab
import argparse
import glob
from torch.utils.data import DataLoader
import time

from pytorch3d.pytorch3d.ops import knn_points, knn_gather
# from pytorch3d.ops import knn_points, knn_gather
# from dist import chamfer_loss, EMD
from dist import chamfer_distance_numpy, chamfer_distance_sklearn
from fusion import *

class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        
        self.conv = nn.Sequential(*layers)
    
    def knn_group(self, points1, points2, features2, k):
        '''
        For each point in points1, query kNN points/features in points2/features2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features2: [B,C,N]
        Output:
            new_features: [B,4,N]
            nn: [B,3,N]
            grouped_features: [B,C,N]
        '''
        # points1 = points1.permute(0,2,1).contiguous()
        # points2 = points2.permute(0,2,1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1,1,k,1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        grouped_features = knn_gather(features2.permute(0,2,1), nn_idx)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0,3,1,2).contiguous(),\
            nn.permute(0,3,1,2).contiguous(),\
            grouped_features.permute(0,3,1,2).contiguous()
    
    def forward(self, points1, points2, features1, features2, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
            t: [B], time step in (0,1)
        Output:
            fused_points: [B,3+C,N]
        '''
        N = points1.shape[-1]
        B = points1.shape[0] # batch size

        new_features_list = []
        new_grouped_points_list = []
        new_grouped_features_list = []

        for i in range(B):
            # t1 = t[i]
            t1 = t
            # new_points1 = points1[i:i+1,:,:]
            # new_points2 = points2[i:i+1,:,:]
            # new_features1 = features1[i:i+1,:,:]
            # new_features2 = features2[i:i+1,:,:]
            new_points1 = torch.Tensor(points1[i:i+1,:])
            new_points2 = torch.Tensor(points2[i:i+1,:])
            new_features1 = features1[i:i+1,:]
            new_features2 = features2[i:i+1,:]

            N2 = int(N*t1)
            N1 = N - N2

            k2 = int(k*t1)
            k1 = k - k2

            randidx1 = torch.randperm(N)[:N1]
            randidx2 = torch.randperm(N)[:N2]
            new_points = torch.cat((new_points1[:,randidx1], new_points2[:,randidx2]), dim=-1)

            new_features1, grouped_points1, grouped_features1 = self.knn_group(new_points, new_points1, new_features1, k1)
            new_features2, grouped_points2, grouped_features2 = self.knn_group(new_points, new_points2, new_features2, k2)

            new_features = torch.cat((new_features1, new_features2), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)
            new_grouped_features = torch.cat((grouped_features1, grouped_features2), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)
            new_grouped_features_list.append(new_grouped_features)

        new_features = torch.cat(new_features_list, dim=0)
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)
        new_grouped_features = torch.cat(new_grouped_features_list, dim=0)

        new_features = self.conv(new_features)
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]
        weights = F.softmax(new_features, dim=-1)

        C = features1.shape[1]
        weights = weights.unsqueeze(1).repeat(1,3+C,1,1)
        fused_points = torch.cat([new_grouped_points, new_grouped_features], dim=1)
        fused_points = torch.sum(torch.mul(weights, fused_points), dim=-1, keepdim=False)

        return fused_points


parser = argparse.ArgumentParser()
parser.add_argument('--filename1', default='E:/fyp/dataset/output/000010.npz', help='Filename to visualize [default: kitti_flow/000012.npz]')
parser.add_argument('--filename2', default='E:/fyp/dataset/output/000010_back.npz', help='Filename to visualize [default: kitti_flow/000012.npz]')

# python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy
parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
parser.add_argument('--model', type=str, default='dcp', metavar='N',
                    choices=['dcp'], help='Model to use, [dcp]')
parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'], help='Embedding nn to use, [pointnet, dgcnn]')
parser.add_argument('--pointer', type=str, default='identity', metavar='N',
                    choices=['identity', 'transformer'], help='Attention-based pointer generator to use, [identity, transformer]')
parser.add_argument('--head', type=str, default='svd', metavar='N',
                    choices=['mlp', 'svd', ], help='Head to use, [mlp, svd]')
parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multiheadedattention')
parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
parser.add_argument('--dropout', type=float, default=0.0, metavar='N', help='Dropout ratio in transformer')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size', help='Size of batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train ')
parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
parser.add_argument('--eval', action='store_true', default=True, help='evaluate the model')
parser.add_argument('--cycle', type=bool, default=False, metavar='N', help='Whether to use cycle consistency')
parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N', help='Wheter to add gaussian noise')
parser.add_argument('--unseen', type=bool, default=False, metavar='N', help='Wheter to test on unseen category')
parser.add_argument('--num_points', type=int, default=1024, metavar='N', help='Num of points to use')
parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N', help='dataset to use')
parser.add_argument('--factor', type=float, default=4, metavar='N', help='Divided factor for rotations')
# parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')

FLAGS = parser.parse_args()

data1 = np.load(FLAGS.filename1)
data2 = np.load(FLAGS.filename2)

# E:\fyp\Point-Cloud-Interpolation-Based-on-Deep-Learning\kitti_flow\000002_back.npz
flow_files = sorted(glob.glob('kitti_flow/*.npz'))
cd_avg=0
cd1_avg=0
cd2_avg=0
for i in range(len(flow_files)): 
    
    data1 = np.load(flow_files[2*i])
    data2 = np.load(flow_files[2*i+1])

    points_mid = data1['points_mid']
    points1 = data1['points1']
    points2 = data1['points2']
    points3 = data1['points3']
    points4 = data1['points4']
    flow = data1['flow']
    mask = data1['mask']


    points1_back = data2['points1']
    points2_back = data2['points2']
    points3_back = data2['points3']
    points4_back = data2['points4']
    flow_back = data2['flow']
    mask_back = data2['mask']

    point_size = 0.1

    # mlab.figure(bgcolor=(1,1,1))
    # mlab.points3d(points1[:, 0], points1[:, 1], points1[:, 2], color=(1, 0, 0), scale_factor=point_size)
    # mlab.points3d(points2[:, 0], points2[:, 1], points2[:, 2], color=(1, 1, 0), scale_factor=point_size)
    # mlab.points3d(points3[:, 0], points3[:, 1], points3[:, 2], color=(0, 1, 0), scale_factor=point_size)
    # mlab.points3d(points4[:, 0], points4[:, 1], points4[:, 2], color=(0, 0, 1), scale_factor=point_size)
    # flowed = points1 + flow
    # mlab.points3d(flowed[:, 0], flowed[:, 1], flowed[:, 2], color=(0, 0, 0), scale_factor=point_size)

    # mlab.view()
    # input()

    ini_pc = points1
    mid_pc = points_mid
    end_pc = points2
    # ini_pc = ini_pc.cuda(non_blocking=True)
    # mid_pc = mid_pc.cuda(non_blocking=True)
    # end_pc = end_pc.cuda(non_blocking=True)
    # t = t.cuda().float()

    warped_points1_xyz = points1 + flow * 0.5
    warped_points2_xyz = points1_back + flow_back * 0.5

    pointcloud1 = torch.Tensor(warped_points1_xyz.T)
    # print('warped_points1_xyz.size(): ',pointcloud1.size())
    pointcloud2 = torch.Tensor(warped_points2_xyz.T)
    # print('warped_points2_xyz.size(): ',pointcloud2.size())
    num_point1 = pointcloud1.size(1)
    num_point2 = pointcloud2.size(1)
    num_point =  num_point1 if num_point1 < num_point2 else num_point2

    pointcloud1=DataLoader([warped_points1_xyz[:num_point].T],batch_size=1, shuffle=True, drop_last=True)
    pointcloud2=DataLoader([warped_points2_xyz[:num_point].T],batch_size=1, shuffle=True, drop_last=True)


    # k = 32

    # ini_color = np.zeros(warped_points1_xyz.shape).astype('float32')
    # ini_color = torch.from_numpy(ini_color).t()
    # end_color = ini_color

    # Points fusion
    # fusion = PointsFusion(4, [64, 64, 128])
    # fused_points = fusion(warped_points1_xyz, warped_points2_xyz, ini_color, end_color, k, 0.5)

    net = DCP(FLAGS).cuda()
    model_path = 'dcp_model/dcp_v1.t7'
    if not os.path.exists(model_path):
        print("can't find pretrained model")
    net.load_state_dict(torch.load(model_path), strict=False)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # get the rigid transformation matrix 
    mse_ab, mae_ab, mse_ba, mae_ba, rotation_ab_pred, translation_ab_pred, \
           rotation_ba_pred, translation_ba_pred, rotation_loss, translation_loss, cycle_loss = get_trans_matrix(net,pointcloud1, pointcloud2)
    
    
    translation_ab_pred = torch.where(abs(translation_ab_pred)>1, translation_ab_pred/10, translation_ab_pred)
    translation_ab_pred = translation_ab_pred/10
    translation_ba_pred = torch.where(abs(translation_ba_pred)>1, translation_ba_pred/10, translation_ba_pred)
    translation_ba_pred = translation_ba_pred/10
    
    # [cos, -sin, 0.],
    # [sin, cos, 0.],
    # [0., 0., 1]
    rotation_ab_pred=rotation_ab_pred.cpu().detach().numpy()
    rotation_ba_pred=rotation_ba_pred.cpu().detach().numpy()

    rot = torch.tensor([[[np.cos(np.arccos(rotation_ab_pred[0][0][0])/2), -np.sin(np.arcsin(-rotation_ab_pred[0][0][1])/2), 0.],
         [np.sin(np.arcsin(rotation_ab_pred[0][1][0])/2), np.cos(np.arccos(rotation_ab_pred[0][1][1])/2), 0.],
         [0., 0., 1.]]])
    rot_ba = torch.tensor([[[np.cos(np.arccos(rotation_ba_pred[0][0][0])/2), -np.sin(np.arcsin(-rotation_ba_pred[0][0][1])/2), 0.],
         [np.sin(np.arcsin(rotation_ba_pred[0][1][0])/2), np.cos(np.arccos(rotation_ba_pred[0][1][1])/2), 0.],
         [0., 0., 1.]]])

    print('the rigid transformation matrix is: ', rot, translation_ab_pred*0.5)
    # print('the rigid transformation matrix is: ', rot_ba, translation_ba_pred*0.5)
    fused_pc = transform_point_cloud(next(iter(pointcloud1)), rot, translation_ab_pred*0.5)
    # fused_pc = transform_point_cloud(next(iter(pointcloud2)), rot_ba, translation_ba_pred*0.5)


    #     chamfer_dist, _ = chamfer_distance(pc1, pc2)

    # cd = chamfer_loss(fused_points[:,:3,:], mid_pc[:,:3,:])
    cd = chamfer_distance_sklearn(fused_pc[0].T.cpu().detach().numpy(), mid_pc)
    cd1 = chamfer_distance_sklearn(warped_points1_xyz, mid_pc)
    cd2 = chamfer_distance_sklearn(warped_points2_xyz, mid_pc)
    # emd = EMD(fused_points[:,:3,:], mid_pc[:,:3,:])

    # print('CD:{:.3} EMD:{:.3}'.format(cd, emd))
    cd_avg+=cd
    cd1_avg+=cd1
    cd2_avg+=cd2

    print('this CD:{:.3}, CD1:{:.3}, CD2:{:.3}'.format(cd, cd1, cd2))
    print('avg CD:{:.3}, CD1:{:.3}, CD2:{:.3}'.format(cd_avg/(i+1), cd1_avg/(i+1), cd2_avg/(i+1)))
    torch.cuda.empty_cache()
    mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(mid_pc[:, 0], mid_pc[:, 1], mid_pc[:, 2], color=(0, 1, 0), scale_factor=point_size)
    mlab.points3d(fused_pc[0].T.cpu().detach().numpy()[:, 0], fused_pc[0].T.cpu().detach().numpy()[:, 1], fused_pc[0].T.cpu().detach().numpy()[:, 2], color=(1, 0, 0), scale_factor=point_size)
    # mlab.points3d(warped_points2_xyz[:, 0], warped_points2_xyz[:, 1], warped_points2_xyz[:, 2], color=(1, 0, 0), scale_factor=point_size)
    mlab.view()
    input()
    time.sleep(30)

