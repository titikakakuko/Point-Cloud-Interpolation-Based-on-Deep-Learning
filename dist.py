import torch
from torch import linalg as LA
# import tensorflow as tf
import numpy as np
from sklearn.neighbors import KDTree
# from pytorch3d.loss import chamfer_distance
# import emd_cuda

# sklearn version
def chamfer_distance_sklearn(array1,array2):
    # batch_size, num_point = array1.shape[:2]
    num_point, batch_size  = array1.shape
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i].reshape(-1, 1), leaf_size=num_point+1)
        tree2 = KDTree(array2[i].reshape(-1, 1), leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i].reshape(-1, 1))
        distances2, _ = tree2.query(array1[i].reshape(-1, 1))
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist

# np version
def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = LA.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    num_point, batch_size  = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist

# # tf version
# def distance_matrix(array1, array2):
#     """
#     arguments: 
#         array1: the array, size: (num_point, num_feature)
#         array2: the samples, size: (num_point, num_feature)
#     returns:
#         distances: each entry is the distance from a sample to array1
#             , it's size: (num_point, num_point)
#     """
#     num_point, num_features = array1.shape
#     expanded_array1 = tf.tile(array1, (num_point, 1))
#     expanded_array2 = tf.reshape(
#             tf.tile(tf.expand_dims(array2, 1), 
#                     (1, num_point, 1)),
#             (-1, num_features))
#     distances = tf.norm(expanded_array1-expanded_array2, axis=1)
#     distances = tf.reshape(distances, (num_point, num_point))
#     return distances

# def av_dist(array1, array2):
#     """
#     arguments:
#         array1, array2: both size: (num_points, num_feature)
#     returns:
#         distances: size: (1,)
#     """
#     distances = distance_matrix(array1, array2)
#     distances = tf.reduce_min(distances, axis=1)
#     distances = tf.reduce_mean(distances)
#     return distances

# def av_dist_sum(arrays):
#     """
#     arguments:
#         arrays: array1, array2
#     returns:
#         sum of av_dist(array1, array2) and av_dist(array2, array1)
#     """
#     array1, array2 = arrays
#     av_dist1 = av_dist(array1, array2)
#     av_dist2 = av_dist(array2, array1)
#     return av_dist1+av_dist2

# def chamfer_distance_tf(array1, array2):
#     batch_size, num_point, num_features = array1.shape
#     dist = tf.reduce_mean(
#                tf.map_fn(av_dist_sum, elems=(array1, array2), dtype=tf.float64)
#            )
#     return dist

# # pytorch3d version
# def chamfer_loss(pc1, pc2):
#     '''
#     Input:
#         pc1: [B,3,N]
#         pc2: [B,3,N]
#     '''
#     pc1 = pc1.permute(0,2,1)
#     pc2 = pc2.permute(0,2,1)
#     chamfer_dist, _ = chamfer_distance(pc1, pc2)
#     return chamfer_dist

# class EarthMoverDistanceFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         xyz1 = xyz1.contiguous()
#         xyz2 = xyz2.contiguous()
#         assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
#         match = emd_cuda.approxmatch_forward(xyz1, xyz2)
#         cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
#         ctx.save_for_backward(xyz1, xyz2, match)
#         return cost

#     @staticmethod
#     def backward(ctx, grad_cost):
#         xyz1, xyz2, match = ctx.saved_tensors
#         grad_cost = grad_cost.contiguous()
#         grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
#         return grad_xyz1, grad_xyz2

# def earth_mover_distance(xyz1, xyz2, transpose=True):
#     """Earth Mover Distance (Approx)

#     Args:
#         xyz1 (torch.Tensor): (b, 3, n1)
#         xyz2 (torch.Tensor): (b, 3, n1)
#         transpose (bool): whether to transpose inputs as it might be BCN format.
#             Extensions only support BNC format.

#     Returns:
#         cost (torch.Tensor): (b)

#     """
#     if xyz1.dim() == 2:
#         xyz1 = xyz1.unsqueeze(0)
#     if xyz2.dim() == 2:
#         xyz2 = xyz2.unsqueeze(0)
#     if transpose:
#         xyz1 = xyz1.transpose(1, 2)
#         xyz2 = xyz2.transpose(1, 2)
#     cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
#     return cost

# def EMD(pc1, pc2):
#     '''
#     Input:
#         pc1: [1,3,M]
#         pc2: [1,3,M]
#     Ret:
#         d: torch.float32
#     '''
#     pc1 = pc1.permute(0,2,1).contiguous()
#     pc2 = pc2.permute(0,2,1).contiguous()
#     d = earth_mover_distance(pc1, pc2, transpose=False)
#     d = torch.mean(d)/pc1.shape[1]
#     return d