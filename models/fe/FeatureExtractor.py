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

class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError

class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict

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
    
from functools import partial
import spconv

class VoxelBackBone8x(nn.Module):
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
        block = post_act_block

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

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

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
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


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
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
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

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

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
        keypoints = self.get_sampled_points(batch_dict)

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=2)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict




class FeatureExtractor(nn.Module):
    default_config = {
        'descriptor_dim': 144,
        'keypoint_encoder': [32, 64, 128],
        'descritor_encoder': [64, 144],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'resolving_distance': 50
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        

        self.vfe = MeanVFE()

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        
        kpts0, kpts1 = data['keypoints0'].double(), data['keypoints1'].double()
        lrf0, lrf1 = data['lrf0'].double(), data['lrf1'].double()
        pc0, pc1 = data['cloud0'], data['cloud1']

        batch_size, _, _ = pc0.size()

        x = time.time()
         
        fps_path = 'fps/{}'.format(data['sequence'][0])
        if not os.path.exists(fps_path):
            os.makedirs(fps_path) 
        
        fps_txt = '{}/{:0>6d}.bin'.format(fps_path,data['idx0'].cpu().numpy()[0])
        if not os.path.exists(fps_txt):
            cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                pc0.contiguous(), 2048
            ).long()[0]

            batch_idx = torch.arange(0, 1, dtype=int).view(-1, 1).repeat(1, 2048)
            pc0 = pc0.reshape((1,-1,8))[:,:,:3]

            indx = cur_pt_idxs.cpu().numpy()
            fp = pc0[batch_idx,indx,:]
            fp.cpu().numpy().astype(np.float32).tofile(fps_txt)
   
            # point_cloud_o3d2 = o3d.geometry.PointCloud()
            # point_cloud_o3d2.points = o3d.utility.Vector3dVector(fp.cpu().numpy()[0,:,:])
            # point_cloud_o3d2.translate(np.asarray([0, 120, 0]))

            # point_cloud_o3d = o3d.geometry.PointCloud()
            # point_cloud_o3d.points = o3d.utility.Vector3dVector(pc0.cpu().numpy()[0, :, :3])
            # o3d.visualization.draw_geometries([point_cloud_o3d+point_cloud_o3d2])

            # pc1 = np.fromfile('/home/chenghao/DL_workspace/Localization/fps/06/000137.bin', dtype=np.float32)
            # pc1 = pc1.reshape((-1, 3))

            # point_cloud_o3d2 = o3d.geometry.PointCloud()
            # point_cloud_o3d2.points = o3d.utility.Vector3dVector(np.asarray(pc1))
            # point_cloud_o3d2.translate(np.asarray([0, 120, 0]))
            # o3d.visualization.draw_geometries([point_cloud_o3d2])
        
            # indx = cur_pt_idxs.cpu().numpy()
            # np.savetxt(fps_txt,indx.astype(int))
        
        fps2_txt = '{}/{:0>6d}.bin'.format(fps_path,data['idx1'].cpu().numpy()[0])
        if not os.path.exists(fps2_txt):
            cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                pc1.contiguous(), 2048
            ).long()[0]

            batch_idx = torch.arange(0, 1, dtype=int).view(-1, 1).repeat(1, 2048)
            pc1 = pc1.reshape((1,-1,8))[:,:,:3]

            indx = cur_pt_idxs.cpu().numpy()
            fp = pc1[batch_idx,indx,:]
            fp.cpu().numpy().astype(np.float32).tofile(fps2_txt)

            # point_cloud_o3d2 = o3d.geometry.PointCloud()
            # point_cloud_o3d2.points = o3d.utility.Vector3dVector(fp.cpu().numpy()[0,:,:])
            # point_cloud_o3d2.translate(np.asarray([0, 120, 0]))

            # point_cloud_o3d = o3d.geometry.PointCloud()
            # point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1.cpu().numpy()[0, :, :3])
            # o3d.visualization.draw_geometries([point_cloud_o3d+point_cloud_o3d2])

            # fp = pc1[batch_idx,indx,:]
            # output = fp.astype(np.float32)
            # output.tofile(fps2_txt)

           
        
            # indx = cur_pt_idxs.cpu().numpy()
            # np.savetxt(fps2_txt,indx.astype(int))
        
        # # np.loadtxt('{}/{:0>6d}'.format(fps_path,data['idx0'].cpu().numpy()[0]))

        return None

        # batch_idx = torch.arange(0, batch_size, dtype=int).view(-1, 1).repeat(1, 2048)
        # fp = pc0[batch_idx, cur_pt_idxs, :]
        # y= time.time()
        # print('pybind ',y-x)

        # point_cloud_o3d2 = o3d.geometry.PointCloud()
        # point_cloud_o3d2.points = o3d.utility.Vector3dVector(fp.numpy()[0,:,:])
        # point_cloud_o3d2.translate(np.asarray([0, 120, 0]))

        # point_cloud_o3d = o3d.geometry.PointCloud()
        # point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1.numpy()[0, :, :3])
        # # point_cloud_o3d.normals = o3d.utility.Vector3dVector(pc1[:, 3:6])
        # o3d.visualization.draw_geometries([point_cloud_o3d+point_cloud_o3d2])
    
        # if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
        #     shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
        #     return {
        #         'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
        #         'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
        #         'matching_scores0': kpts0.new_zeros(shape0)[0],
        #         'matching_scores1': kpts1.new_zeros(shape1)[0],
        #         'skip_train': True
        #     }
        # # file_name = data['file_name']
        
        # # Keypoint normalization.
        # # kpts0 = normalize_keypoints(kpts0, data['cloud0'].shape)
        # # kpts1 = normalize_keypoints(kpts1, data['cloud1'].shape)

        # if self.descriptor == 'FPFH':
        #     desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
        #     # Keypoint MLP encoder.
        #     desc0 = self.denc(desc0) + self.kenc(kpts0, data['scores0'])
        #     desc1 = self.denc(desc1) + self.kenc(kpts1, data['scores1'])
        #     # Multi-layer Transformer network.
        #     desc0, desc1 = self.gnn(desc0, desc1, kpts0, kpts1, lrf0, lrf1, self.k, self.config['L'])
        #     # Final MLP projection.
        #     mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # else:
        #     raise Exception('Invalid descriptor.')

        # # Compute matching descriptor distance.
        # scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        # scores = scores / self.config['descriptor_dim']**.5

        # # Run the optimal transport.
        # scores = log_optimal_transport(
        #     scores, self.bin_score,
        #     iters=self.config['sinkhorn_iterations'])

        # gt_matches0 = data['match0'] # shape=torch.Size([1, 87, 2])
        # gt_matches1 = data['match1'] # shape=torch.Size([1, 87, 2])

        # # 输出match结果
        # if self.loss_method == 'superglue':
        #     max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        #     indices0, indices1 = max0.indices, max1.indices
        #     zero = scores.new_tensor(0)
        #     if self.mutual_check:
        #         mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0) #gather 沿给定轴dim，将输入索引张量index指定位置的值进行聚合。
        #         mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        #         mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        #         mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        #         valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        #         valid1 = mutual1 & valid0.gather(1, indices1)
        #     else:
        #         valid0 = max0.values.exp() > self.config['match_threshold']
        #         valid1 = max1.values.exp() > self.config['match_threshold']
        #         mscores0 = torch.where(valid0, max0.values.exp(), zero)
        #         mscores1 = torch.where(valid1, max1.values.exp(), zero)
        # else:
        #     max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
        #     indices0, indices1 = max0.indices, max1.indices
        #     valid0, valid1 = indices0<(scores.size(2)-1), indices1<(scores.size(1)-1)
        #     zero = scores.new_tensor(0)
        #     if valid0.sum() == 0:
        #         mscores0 = torch.zeros_like(indices0, device='cuda')
        #         mscores1 = torch.zeros_like(indices1, device='cuda')
        #     else:
        #         if self.mutual_check:
        #             batch = indices0.size(0)
        #             a0 = arange_like(indices0, 1)[None][valid0].view(batch,-1) == indices1.gather(1, indices0[valid0].view(batch,-1))
        #             a1 = arange_like(indices1, 1)[None][valid1].view(batch,-1) == indices0.gather(1, indices1[valid1].view(batch,-1))
        #             mutual0 = torch.zeros_like(indices0, device='cuda') > 0
        #             mutual1 = torch.zeros_like(indices1, device='cuda') > 0
        #             mutual0[valid0] = a0
        #             mutual1[valid1] = a1
        #             mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        #             mscores1 = torch.where(mutual1, max1.values.exp(), zero)
        #         else:
        #             mscores0 = torch.where(valid0, max0.values.exp(), zero)
        #             mscores1 = torch.where(valid1, max1.values.exp(), zero)
        # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))


        
        # # calculate loss
        # if self.loss_method == 'superglue':
        #     loss = superglue()
        # elif self.loss_method == 'triplet_loss':
        #     loss = triplet(self.triplet_loss_gamma)
        # elif self.loss_method == 'gap_loss':
        #     loss = gap(self.triplet_loss_gamma)
        # elif self.loss_method == 'gap_loss_plus':
        #     loss_mean = gap_plus(gt_matches0, gt_matches1, scores, self.triplet_loss_gamma, self.config['var_weight'])
        # elif self.loss_method == 'gap_loss_plusplus':
        #     loss_mean = gap_plusplus(gt_matches0, gt_matches1, scores, self.triplet_loss_gamma, self.config['var_weight'])
        # elif self.loss_method == 'distribution_loss':
        #     loss = distribution(self.triplet_loss_gamma)
        # elif self.loss_method == 'distribution_loss4':
        #     loss = distribution4(self.triplet_loss_gamma)
        # elif self.loss_method == 'distribution_loss6':
        #     loss = distribution6(self.triplet_loss_gamma, self.lamda)
        # elif self.loss_method == 'distribution_loss7':
        #     loss = distribution7(self.triplet_loss_gamma)
        # elif self.loss_method == 'distribution_loss8':
        #     loss = distribution8(self.triplet_loss_gamma)
        
        # if torch.cuda.is_available():
        #     device=torch.device('cuda:{}'.format(self.local_rank[0]))
        #     if torch.cuda.device_count() > 1:
        #         loss = torch.nn.DataParallel(loss, device_ids=self.local_rank)
        #     else:
        #         loss = torch.nn.DataParallel(loss)
        # else:
        #     device = torch.device("cpu")
        # loss.to(device)
        
        # if self.loss_method == 'distribution_loss6':
        #     b,d,n = mdesc0.size()
        #     _,_,m = mdesc1.size()
        #     distance = mdesc0[:,:,:,None].expand(b,d,n,m) - mdesc1[:,:,None].expand(b,d,n,m)
        #     distance = torch.sqrt(torch.sum(distance**2, 1)/d)
        #     loss_mean = loss(gt_matches0, gt_matches1, scores, distance)
        # else:
        #     loss_mean = loss(gt_matches0, gt_matches1, scores)

        # return {
        #     'matches0': indices0, # use -1 for invalid match
        #     'matches1': indices1, # use -1 for invalid match
        #     'matching_scores0': mscores0,
        #     'matching_scores1': mscores1,
        #     'loss': loss_mean,
        #     # 'skip_train': False
        # }

    def update_lamda(self, epoch, indicator):
        lamda_clip = 1
        self.lamda = 0.02*(epoch-indicator) + self.lamda_initial

        if self.lamda > lamda_clip:
            self.lamda = lamda_clip

    # def update_lamda(self, epoch, indicator):
    #     lamda_clip = 1

    #     self.lamda = lamda_clip
    #     print('update lamda')