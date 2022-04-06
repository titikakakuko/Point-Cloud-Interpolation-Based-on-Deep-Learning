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

# from point_4d_convolution import *
# from transformer import *
from dcp import *




class P4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output


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

def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


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

def test(args, net, test_loader):

    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)
    test_rmse_ba = np.sqrt(test_mse_ba)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
    test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    print('==FINAL TEST==')
    print('A--------->B')
    print('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
                     test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    print('B--------->A')
    print('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                     test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

# parser = argparse.ArgumentParser(description='Point Cloud Registration')
# parser.add_argument('--cycle', type=bool, default=False, metavar='N',
#                         help='Whether to use cycle consistency')

# args = parser.parse_args()
# net = DCP(args).cuda()

# model_path = 'checkpoints' + '/' + '/models/model.best.t7'
# if not os.path.exists(model_path):
#     print("can't find pretrained model")
# net.load_state_dict(torch.load(model_path), strict=False)
# if torch.cuda.device_count() > 1:
#     net = nn.DataParallel(net)
#     print("Let's use", torch.cuda.device_count(), "GPUs!")

# # test(args, net, test_loader)
# get_trans_matrix(net,warped_points1_xyz, warped_points2_xyz)


