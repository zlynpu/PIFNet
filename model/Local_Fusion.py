import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
import numpy as np
import sys
import os

ROOT_DIR = os.path.abspath('/data1/zhangliyuan/code/IMFNet_exp')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

def local_fusion(coord, feature, image_feature, image_shape, extrinsic, intrinsic,voxel_size=0.025):
    # batch ----- batch
    lengths = []
    max_batch = torch.max(coord[:, 0])
    for i in range(max_batch + 1):
        length = torch.sum(coord[:, 0] == i)
        lengths.append(length)

    device = torch.device('cuda')
    fusion_feature_batch = []
    start = 0
    end = 0
    batch_id = 0
    
    for length in lengths:
        end += length
        point_coord = coord[start:end, 1:] * voxel_size
        point_feature = feature[start:end, :].unsqueeze(0)
        point_feature = point_feature.permute(0,2,1)

        point_z = point_coord[:, -1]
        # print('pointcoord',point_coord.shape)
        one = torch.ones((point_coord.shape[0],1)).to(device)
        # print('one',one.shape)
        point_in_lidar = torch.cat([point_coord, one],1).t()
        # print('point_in_lidar',point_in_lidar.shape)

        point_in_camera = extrinsic[batch_id, :, :].mm(point_in_lidar)
        point_in_image = intrinsic[batch_id, :, :].mm(point_in_camera)/point_z
        point_in_image = point_in_image.t()
        point_in_image[:, -1] = point_z

        point_in_image[:, 0] = point_in_image[:, 0] * 2 / image_shape[batch_id, 0] - 1
        point_in_image[:, 1] = point_in_image[:, 1] * 2 / image_shape[batch_id, 1] - 1

        point_in_image = point_in_image.unsqueeze(0)

        feature_map = image_feature[batch_id, :, :, :].unsqueeze(0)
        # print('feature_map_shape:',feature_map.shape)
        xy = point_in_image[:, :, :-1].unsqueeze(1)
        # print('xy:',xy.shape)
        img_feature = grid_sample(feature_map, xy)
        img_feature = img_feature.squeeze(2)
        
        
        fusion_feature = torch.cat([img_feature, point_feature], dim=1)
        # print('fusion_feature:', fusion_feature.shape)

        fusion_feature_batch.append(fusion_feature)
        start += length
        batch_id += 1
    
    fusion_feature_batch = torch.cat(fusion_feature_batch,2)
    # print('fusion_feature_batch:',fusion_feature_batch.shape)
    return fusion_feature_batch



