from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import time
from models.pointnet.pointnet_util import PointNetSetKptsMsg, PointNetSetAbstraction

import numpy as np
from utils.utils_loss import (superglue, triplet, gap, gap_plus, 
                                gap_plusplus, distribution, 
                                distribution2, distribution4, distribution6, distribution7, distribution8)


from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

import open3d as o3d 
import os

from utils import common_utils
from functools import partial
import spconv

import sparseconvnet as scn
import torch.nn.functional as F

class MeanVFE(nn.Module):
    def __init__(self):
        super().__init__()

    # def get_output_feature_dim(self):
    #     return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (batch, num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points[:,None], min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict

class VoxelBackBone8x(nn.Module):
    '''spconv version. Still exists bug.'''
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        
        
        self.conv1 = scn.Sequential(
            scn.SubmanifoldConvolution(3, 4, 16, 3, False),
            scn.BatchNormReLU(16),
            scn.SubmanifoldConvolution(3, 16, 16, 3, False),
            scn.BatchNormReLU(16)
        )

        self.conv2 = scn.Sequential(
            scn.Convolution(3, 16, 32, 3, 2, False),
            scn.BatchNormReLU(32),
            # scn.SubmanifoldConvolution(3, 32, 32, 3, False),
            # scn.BatchNormReLU(32),
            scn.SubmanifoldConvolution(3, 32, 32, 3, False),
            scn.BatchNormReLU(32)
        )

        self.conv3 = scn.Sequential(
            scn.Convolution(3, 32, 64, 3, 2, False),
            scn.BatchNormReLU(64),
            # scn.SubmanifoldConvolution(3, 64, 64, 3, False),
            # scn.BatchNormReLU(64),
            scn.SubmanifoldConvolution(3, 64, 64, 3, False),
            scn.BatchNormReLU(64)
        )

        self.conv4 = scn.Sequential(
            scn.Convolution(3, 64, 64, 3, 2, False),
            scn.BatchNormReLU(64),
            # scn.SubmanifoldConvolution(3, 64, 64, 3, False),
            # scn.BatchNormReLU(64),
            scn.SubmanifoldConvolution(3, 64, 64, 3, False),
            scn.BatchNormReLU(64)
        )

        self.conv_out = scn.Sequential(
            scn.Convolution(3, 64, 128, [3, 1, 1], 1, False),
            scn.BatchNormReLU(128)
        )

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        inputSpatialSize = self.conv1.input_spatial_size(torch.LongTensor(self.sparse_shape))
        self.input_layer = scn.InputLayer(3, inputSpatialSize)

    

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
    
    def Padding(self, input_layer, padding=1):
        features = input_layer.features
        pad = nn.ConstantPad2d((0,1,0,0),0)
        return features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """        
        input = self.input_layer([batch_dict['voxel_coords'], batch_dict['voxel_features']])
        # input = self.Padding(input)
        x_conv1 = self.conv1(input)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelBackBone8x_spconv(nn.Module):
    '''spconv version. Still exists bug.'''
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
    
    def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        else:
            raise NotImplementedError

        m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(),
        )

        return m

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        for key, val in batch_dict.items():
            if key in ['voxels', 'voxel_num_points', 'voxel_features']:
                coors = []
                for i, coor in enumerate(val):
                    coors.append(coor)
                batch_dict[key] = torch.cat(coors,dim=0)
                # batch = val.size()[0]
                # batch_dict[key] = val.view(-1, m, d)
                # ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    '''[batch, n, dim]  -->  [n, 1+dim] (batch_idx, x, y, z)'''
                    pad = nn.ConstantPad2d((1,0,0,0),i)
                    coor_pad = pad(coor)
                    # coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                batch_dict[key] = torch.cat(coors,dim=0)

        input_sp_tensor = spconv.SparseConvTensor(
            features=batch_dict['voxel_features'],
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size']
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, NUM_OUTPUT_FEATURES, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.todense = scn.SparseToDense(3, NUM_OUTPUT_FEATURES)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = self.todense(encoded_spconv_tensor)
        # spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules

class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0

        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])
        
        self.mlps = nn.Sequential(
            nn.Linear(3, 8,  bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32,  bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        c_in=1

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

        self.nsample =16
        self.group = pointnet2_utils_stack.QueryAndGroup2(0.4, self.nsample, use_xyz=True)


    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        # keypoints = self.get_sampled_points(batch_dict)
        keypoints = batch_dict['keypoints'][:,:,:3]
        batch_size, num_keypoints, _ = keypoints.shape

        new_xyz = keypoints.view(-1, 3).contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)
        # if torch.isnan(self.SA_rawpoints.mlps[0][0].weight[0]).sum()>0:
        #     print('pause')

        '''group KNN neighbour'''
        # raw_points = batch_dict['points']
        # xyz = raw_points[:, :3]
        # xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        # for bs_idx in range(batch_size):
        #     xyz_batch_cnt[bs_idx] = (raw_points[:, -1] == bs_idx).sum()
        # all_keypoints, all_keypoints_indice = \
        #             self.group(xyz.contiguous(), xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        # all_keypoints = all_keypoints.transpose(1,2).contiguous()
        # all_keypoints = all_keypoints.view(batch_size, num_keypoints, self.nsample, 3)
        # all_keypoints = all_keypoints.reshape(batch_size, num_keypoints*self.nsample, 3)
        
        # keypoints = all_keypoints
        # num_keypoints = num_keypoints * self.nsample
        # new_xyz = keypoints.view(-1, 3).contiguous()
        # new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)


        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

        
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            a = time.time()
            raw_points = batch_dict['points']
            xyz = raw_points[:, :3]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, -1] == bs_idx).sum()
            point_features = raw_points[:, 3:-1].contiguous() if raw_points.shape[1] > 4 else None

            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))
            b= time.time()
            # print('raw point ',b-a)
            print(self.SA_rawpoints.mlps[0][0].weight[0])


        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].get_spatial_locations()
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, :3],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range,
                device = batch_dict['multi_scale_3d_features'][src_name].features.device
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, -1] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))


        

        key_points = batch_dict['keypoints'].view(-1, 4).contiguous()
        point_features = key_points[:,:3].contiguous() 
        point_features = self.mlps(point_features.view(-1, point_features.shape[-1]))
        batch_dict['point_features'] = point_features  # (BxN, C)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)

        # point_features = torch.cat(point_features_list, dim=2)

        # batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        # point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        # batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        # point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        # batch_dict['point_features'] = point_features  # (BxN, C)
        # batch_dict['point_coords'] = point_coords  # (BxN, 4)

        if torch.isnan(point_features).sum()>0:
            print('pause')

        
        return batch_dict




class FeatureExtractor(nn.Module):
    # default_config = {
    #     'descriptor_dim': 144,
    #     'keypoint_encoder': [32, 64, 128],
    #     'descritor_encoder': [64, 144],
    #     'GNN_layers': ['self', 'cross'] * 9,
    #     'sinkhorn_iterations': 100,
    #     'match_threshold': 0.2,
    #     'resolving_distance': 50
    # }

    def __init__(self, cfgs, dataset):
        super().__init__()
        # self.config = {**self.default_config, **config}

        # self.vfe = MeanVFE()
        # self.vconv = VoxelBackBone8x(cfgs.BACKBONE_3D, dataset.num_point_features, dataset.grid_size)
        self.hc = HeightCompression(cfgs.MAP_TO_BEV, cfgs.PFE.NUM_OUTPUT_FEATURES)
        self.pfe = VoxelSetAbstraction(cfgs.PFE, dataset.voxel_size, dataset.point_cloud_range, self.hc.num_bev_features, dataset.num_point_features)
        
        self.point_cloud_range = dataset.point_cloud_range
        self.grid_size = dataset.grid_size
        self.num_point_features = dataset.num_point_features
        self.voxel_size = dataset.voxel_size
        self.max_num_points_per_voxel = dataset.max_num_points_per_voxel
        self.max_voxels = dataset.max_voxels
        self.max_points = dataset.max_points

        self.groupers = pointnet2_utils_stack.QueryAndGroup(1, 50, use_xyz=True)
        self.threshold = dataset.threshold
        self.mutual_check = dataset.mutual_check

        self.num_keypoints = 2048

    def UNPadding(self, bacth_dict):
        '''Handle uneven input'''
        points = {}
        # kpss = {}
        for key in ['pc1', 'pc0']:
            val = bacth_dict[key]
            coors = []
            coors_pad = []
            # kps = []
            for i, coor in enumerate(val):
                coor = coor[:(coor[:,-1]>-1).sum(),:]
                pad = nn.ConstantPad2d((0,1,0,0),i)
                coor_pad = pad(coor)
                coors.append(coor)
                coors_pad.append(coor_pad)

                # cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                #     coor[None,:,:3].contiguous(), self.num_keypoints
                # ).long()[0]
                # kp = coor[cur_pt_idxs,:]
                # kps.append(kp)

            bacth_dict[key] = torch.cat(coors_pad,dim=0)
            points[key] = coors
            # kpss[key] = torch.stack(kps,dim=0)
        
        # voxels_num = []
        # for i, key in enumerate(['voxel_num_points0', 'voxel_num_points1']):
        #     val = bacth_dict[key]
        #     voxels_num.append([])
        #     coors = []
        #     for b, coor in enumerate(val):
        #         voxels_num[i].append((coor>0).sum())
        #         coor = coor[:voxels_num[i][b]]
        #         coors.append(coor)
                
        #     bacth_dict[key] = torch.cat(coors,dim=0)
        
        # for i, key in enumerate(['voxel_coords0', 'voxel_coords1']):
        #     val = bacth_dict[key]
        #     coors = []
        #     for b, coor in enumerate(val):
        #         coor = coor[:voxels_num[i][b]]
        #         pad = nn.ConstantPad2d((0,1,0,0),b)
        #         coor = pad(coor)
        #         coors.append(coor)
        #     bacth_dict[key] = torch.cat(coors,dim=0)

        # for i, key in enumerate(['voxels0', 'voxels1']):
        #     val = bacth_dict[key]
        #     coors = []
        #     for b, coor in enumerate(val):
        #         coor = coor[:voxels_num[i][b]]
        #         coors.append(coor)
        #     bacth_dict[key] = torch.cat(coors,dim=0)
                
        return bacth_dict, points
    
    def transform_points_to_voxels(self, points):
        '''torchsparse'''
        # from torchsparse.utils.quantize import sparse_quantize
        # inputs = np.random.uniform(-100, 100, size=(10000, 4))

        # coords, feats = inputs[:, :3], inputs
        # coords -= np.min(coords, axis=0, keepdims=True)
        # coords, indices = sparse_quantize(coords,
        #                                   0.2,
        #                                   return_index=True)
        # coords = torch.tensor(coords, dtype=torch.int)
        # feats = torch.tensor(feats[indices], dtype=torch.float)

        # from torch import nn
        # from torchsparse import SparseTensor
        # from torchsparse import nn as spnn
        # pad = nn.ConstantPad2d((0,1,0,0),0)
        # coords = pad(coords)
        # input_sp_tensor = SparseTensor(coords=coords, feats=feats)
        # net = nn.Sequential(
        #     spnn.Conv3d(4, 16, 3, bias=False),
        #     spnn.BatchNorm(16),
        #     spnn.ReLU(True),
        # )
        # x = net(input_sp_tensor)

        '''spconv 1.2'''
        # import spconv
        # voxel_generator = spconv.utils.VoxelGenerator(
        #     voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range,
        #     max_num_points=self.max_num_points,
        #     max_voxels=self.max_voxels,
        #     # full_mean=False
        # )        
        # # pc = np.random.uniform(-10, 10, size=[1000, 3])
        # voxel_features, voxel_coords, num_points = voxel_generator.generate(points.astype(np.float32))
        # voxel_features = voxel_features[:,0,:]
        # voxel_coords = np.pad(voxel_coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        # voxel_coords = torch.tensor(voxel_coords, dtype=torch.int32)
        # voxel_features = torch.tensor(voxel_features, dtype=torch.float32)
        # # self.shape = [80, 200, 200]
        # x = spconv.SparseConvTensor(voxel_features, voxel_coords, self.grid_size, 1)
        
        
        '''spconv 2'''
        # from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
        # # this generator generate ZYX indices.
        # gen = PointToVoxel(
        #     vsize_xyz=[0.1, 0.1, 0.1], 
        #     coors_range_xyz=[-80, -80, -2, 80, 80, 6], 
        #     num_point_features=3, 
        #     max_num_voxels=5000, 
        #     max_num_points_per_voxel=5)
        # pc = np.random.uniform(-10, 10, size=[1000, 3])
        # pc_th = torch.from_numpy(pc)
        # voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)
    
        from spconv.pytorch.utils import PointToVoxel
        voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel=self.max_num_points_per_voxel,
            max_num_voxels=self.max_voxels,
            device=points.device
        )        
        voxel_output = voxel_generator(points)
        voxels, coordinates, num_points = voxel_output

        # data_dict={}
        
        # if not data_dict['use_lead_xyz']:
        #     voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        # data_dict['points'] = points
        return voxels, coordinates, num_points

    def Voxelization(self, points, batch_size):
        voxels0 = []; voxel_coords0 = []; voxel_num_points0 = []
        voxels1 = []; voxel_coords1 = []; voxel_num_points1 = [] 
        for i in range(batch_size):
            pc0, pc1 = points['pc0'][i], points['pc1'][i]
            a0, b0, c0 = self.transform_points_to_voxels(pc0)
            a1, b1, c1 = self.transform_points_to_voxels(pc1)
            pad = nn.ConstantPad2d((0,1,0,0),i)
            b0 = pad(b0)
            b1 = pad(b1)
            voxels0.append(a0)
            voxel_coords0.append(b0)
            voxel_num_points0.append(c0)
            voxels1.append(a1)
            voxel_coords1.append(b1)
            voxel_num_points1.append(c1)
        voxels0 = torch.cat(voxels0,dim=0)
        voxel_coords0 = torch.cat(voxel_coords0,dim=0)
        voxel_num_points0 = torch.cat(voxel_num_points0,dim=0)
        voxels1 = torch.cat(voxels1,dim=0)
        voxel_coords1 = torch.cat(voxel_coords1,dim=0)
        voxel_num_points1 = torch.cat(voxel_num_points1,dim=0)

        batch0 = {
            'voxels': voxels0,
            'voxel_coords': voxel_coords0,
            'voxel_num_points': voxel_num_points0
        }

        batch1 = {
            'voxels': voxels1,
            'voxel_coords': voxel_coords1,
            'voxel_num_points': voxel_num_points1
        }

        return batch0, batch1
    
    def FPS(self, data, batch_size, device):
        keypoints = {}
        for key in ['rd_pc0', 'rd_pc1']:
            raw_points = data[key]

            # xyz = raw_points[:,:3]
            # xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            # for bs_idx in range(batch_size):
            #     xyz_batch_cnt[bs_idx] = (raw_points[:, -1] == bs_idx).sum()

            # if self.mode == 'train' or self.mode == 'val':
            #     '''Use pre-extracted key point'''
            #     kp_np_file1 = os.path.join(self.train_path,'fps', sequence, '%06d.bin' % (index_in_seq))
            #     kp_np1 = np.fromfile(kp_np_file1, dtype=np.float32)
            #     kp1 = kp_np1.reshape((-1, 4))

            #     kp_np_file2 = os.path.join(self.train_path,'fps', sequence, '%06d.bin' % (index_in_seq2))
            #     kp_np2 = np.fromfile(kp_np_file2, dtype=np.float32)
            #     kp2 = kp_np2.reshape((-1, 4))

            #     kp1 = torch.tensor(kp1, dtype=torch.float, device=device)
            #     kp2 = torch.tensor(kp2, dtype=torch.float, device=device)
            # elif self.mode == 'test':
            '''FPS the key point in real-time'''
            cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                raw_points.contiguous(), self.num_keypoints
            ).long()

            B = torch.arange(0, batch_size, device=device).view(-1, 1).repeat(1, self.num_keypoints)
            kp = raw_points[B, cur_pt_idxs, :]
            keypoints[key] = kp

        return keypoints

    def  CalculateMatches(self, keypoints, data, batch_size, device):
        '''calculate ground true matches'''
        kp0 = keypoints['rd_pc0'][:,:,:3]
        pad = nn.ConstantPad3d((0,1,0,0),1)
        kp1_ = pad(keypoints['rd_pc1'][:,:,:3])
        kp1to0 = torch.einsum('nki,nij->njk', data['T_gt'], kp1_.transpose(1,2))[:,:,:3]
        
        match0to1, match1to0 = -1 * torch.ones((batch_size, len(kp0[0])), dtype=torch.long, device=device), \
                            -1 * torch.ones((batch_size, len(kp1to0[0])), dtype=torch.long, device=device)

        dists = torch.norm((kp0.view(batch_size,len(kp0[0]),1,3)-kp1to0.view(batch_size,1,len(kp1to0[0]),3)),dim=3)  #
        min0to1 = torch.min(dists, dim=2)
        match0to1[min0to1.values < self.threshold] = min0to1.indices[min0to1.values < self.threshold]

        min1to0 = torch.min(dists, dim=1)
        match1to0[min1to0.values < self.threshold] = min1to0.indices[min1to0.values < self.threshold]

        rep = (min0to1.values < self.threshold).sum(dim=1)

        return match0to1, match1to0, rep

    def FindSaliencePoint(self, batch_dict0, batch_dict1, batch_size, device):
        '''Find keypoint amoung neighbours based on salience score'''
        # features0 = batch_dict0['point_features'].view(batch_size, self.num_keypoints, 16, -1)
        # features1 = batch_dict1['point_features'].view(batch_size, self.num_keypoints, 16, -1)
        # coords0 = batch_dict0['point_coords'].view(batch_size, self.num_keypoints, 16, -1)
        # coords1 = batch_dict1['point_coords'].view(batch_size, self.num_keypoints, 16, -1)
        # B = torch.arange(0, batch_size, device=device).view(-1, 1).repeat(1, self.num_keypoints)
        # N = torch.arange(0, self.num_keypoints, device=device).view(1, -1).repeat(batch_size, 1)

        # alpha = torch.exp(features0)/torch.sum(torch.exp(features0),dim=2)[:,:,None]
        # beta = features0/torch.max(features0, dim=3).values[:,:,:,None]
        # salience_score = torch.max(alpha * beta, dim=-1).values
        # indices = torch.max(salience_score, dim=2).indices
        # salience_score1 = torch.max(salience_score, dim=2).values
        # features0 = features0[B,N,indices,:]
        # coords0 = coords0[B,N,indices,1:]

        # alpha = torch.exp(features1)/torch.sum(torch.exp(features1),dim=2)[:,:,None]
        # beta = features1/torch.max(features1, dim=3).values[:,:,:,None]
        # salience_score = torch.max(alpha * beta, dim=-1).values
        # indices = torch.max(salience_score, dim=2).indices
        # salience_score2 = torch.max(salience_score, dim=2).values
        # features1 = features1[B,N,indices,:]
        # coords1 = coords1[B,N,indices,1:]

        # keypoints = {}
        # keypoints['rd_pc0'] = coords0
        # keypoints['rd_pc1'] = coords1

        '''calculate salience score'''
        features0 = batch_dict0['point_features'].view(batch_size, self.num_keypoints, -1)
        features1 = batch_dict1['point_features'].view(batch_size, self.num_keypoints, -1)
        coords0 = batch_dict0['point_coords'].view(batch_size, self.num_keypoints, -1)
        coords1 = batch_dict1['point_coords'].view(batch_size, self.num_keypoints, -1)
        alpha = torch.exp(features0)/torch.sum(torch.exp(features0),dim=1)[:,None]
        beta = features0/torch.max(features0, dim=2).values[:,:,None]
        salience_score1 = torch.max(alpha * beta, dim=-1).values
        alpha = torch.exp(features1)/torch.sum(torch.exp(features1),dim=1)[:,None]
        beta = features1/torch.max(features1, dim=2).values[:,:,None]
        salience_score2 = torch.max(alpha * beta, dim=-1).values
        keypoints = {}
        keypoints['rd_pc0'] = coords0
        keypoints['rd_pc1'] = coords1

        return features0, features1, keypoints, salience_score1, salience_score2

    def EuclideanDistances(self,a,b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=2).unsqueeze(2)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=2).unsqueeze(1)  # n->[1, n]
        bt = b.transpose(2,1)
        return torch.sqrt(torch.clamp(sum_sq_a+sum_sq_b-2*torch.einsum('bnd,bdm->bnm', a, bt), min=0))

    def forward(self, data):
        """Run PVRCNN on keypoints:MEANVFE + VoxelBackBone8x + HeightCompression +
            VoxelSetAbstraction + BaseBEVBackbone
        """
        c = time.time()
        batch_size, _, _ = data['pc0'].size()
        device = data['pc0'].device
        data, points = self.UNPadding(data)
        # keypoints = self.FPS(data, batch_size, device)
        keypoints={}
        keypoints['rd_pc0'] = data['keypoints0']
        keypoints['rd_pc1'] = data['keypoints1']
        # with torch.no_grad():
        # match0to1, match1to0, rep = self.CalculateMatches(keypoints, data, batch_size, device)
        match0to1 = data['match0'].long().cuda()
        a2 = time.time()
        # print('fps ', a2-c)

        # batch_dict0, batch_dict1 = self.Voxelization(points, batch_size)
        batch_dict0={}
        batch_dict1={}
        batch_dict0['batch_size'] = batch_dict1['batch_size'] = batch_size
        batch_dict0['keypoints'], batch_dict1['keypoints'] = keypoints['rd_pc0'], keypoints ['rd_pc1']
        batch_dict0['points'], batch_dict1['points'] = data['pc0'], data['pc1']

        a = time.time()
        # print('voxel ', a-a2)
        # batch_dict0, batch_dict1 = self.vfe(batch_dict0), self.vfe(batch_dict1)
        # batch_dict0, batch_dict1 = self.vconv(batch_dict0), self.vconv(batch_dict1)
        # batch_dict0, batch_dict1 = self.hc(batch_dict0), self.hc(batch_dict1)
        a1 = time.time()
        # print('spconv ', a1-a)
        batch_dict0, batch_dict1 = self.pfe(batch_dict0), self.pfe(batch_dict1)
        b = time.time()
        # print('pfe ', b-a1)

        features0, features1, keypoints, salience_score0, salience_score1 =\
            self.FindSaliencePoint(batch_dict0, batch_dict1, batch_size, device)
        

        feature_distance = self.EuclideanDistances(features0, features1)

        # with torch.no_grad():
        # after_match0to1, after_match1to0, rep2 = self.CalculateMatches(keypoints, data, batch_size, device)
        B = torch.arange(0, batch_size, device=device).view(-1, 1).repeat(1, self.num_keypoints)
        N = torch.arange(0, self.num_keypoints, device=device).view(1, -1).repeat(batch_size, 1)
        after_match0to1 = match0to1
        cdist = feature_distance[B,N,after_match0to1]

        # Or = (match0to1>0)+(after_match0to1>0)
        # And = (match0to1>0)*(after_match0to1>0)
        # Xor_penalty = (Or==True)*(after_match0to1<0)
        # Xor_reward = (Or==True)*(match0to1<0)

        # cdist_with_salience = cdist*(salience_score1[B,after_match0to1]+salience_score0)
        # rep_loss = (cdist_with_salience[Xor_penalty].sum()+1).log() - (cdist_with_salience[Xor_reward].sum()+1).log()
        # feature_loss = (cdist[after_match0to1>0].sum()+1).log()

        # top2 = (-feature_distance).topk(2,dim=-1)
        # a = top2[1][:,:,0]==after_match0to1
        # hard = -top2[0][:,:,0]
        # hard[a] = -top2[0][:,:,1][a]

        # feature_loss = torch.clamp((cdist - hard)+0.5,min=0)

        feature_loss = cdist.log()

        # feature_loss = ((cdist - hard)*(salience_score1[B,after_match0to1]+salience_score0))[after_match0to1>0]
        rep_loss=0

        loss = rep_loss + feature_loss.mean()

        b1 = time.time()
        # print('find salience point ', b1-b)
        # print('rep ',rep)
        # print('rep2 ',rep2)

        if torch.isnan(loss):
            print('pause')

       
        
        return {
            'loss': loss
        }


    def update_lamda(self, epoch, indicator):
        lamda_clip = 1
        self.lamda = 0.02*(epoch-indicator) + self.lamda_initial

        if self.lamda > lamda_clip:
            self.lamda = lamda_clip

    # def update_lamda(self, epoch, indicator):
    #     lamda_clip = 1

    #     self.lamda = lamda_clip
    #     print('update lamda')