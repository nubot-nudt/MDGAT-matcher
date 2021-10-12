# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import time
from models.pointnet.pointnet_util import PointNetSetKptsMsg, PointNetSetAbstraction

import numpy as np
from util.utils_loss import (superglue, triplet, gap, gap_plus, 
                                gap_plusplus, distribution, 
                                distribution2, distribution6, distribution7)

                            
from scipy.spatial.distance import cdist


def knn(x, src, k):
    '''
    Find K nearest neighbour points
    return index
    '''
    inner = -2*torch.matmul(x.transpose(2, 1), src)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    ss = torch.sum(src**2, dim=1, keepdim=True)
    pairwise_distance = -xx.transpose(2, 1) - inner - ss
    # 距离取负，所以topk就是距离最小的k个。[0]返回值，[1]返回index
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, src, k, idx=None):
    '''
    get K nearest neighbour nodes
    return Adjacency Matrix
    '''
    batch_size, num_dims, n = x.size()
    _,           _,       m = src.size()
    x = x.view(batch_size, -1, n)
    src = src.view(batch_size, -1, m)
    if idx is None:
        idx = knn(x, src, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    # sch，邻接矩阵
    A = torch.zeros([batch_size, n, m], dtype=int, device=device)
    B = torch.arange(0, batch_size, device=device).view(-1, 1, 1).repeat(1, n, k)
    N = torch.arange(0, n, device=device).view(1, -1, 1).repeat(batch_size, 1, k)
    A[B,N,idx] = 1

    # 提取邻接feature，十分占内存
    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*m
    # idx = idx + idx_base
    # idx = idx.view(-1)
    # src = src.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # feature = src.view(batch_size*m, -1)[idx, :]
    # feature = feature.view(batch_size, n*k, num_dims)
    # # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # feature = feature.permute(0, 2, 1).contiguous()
    # # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # return feature, A
    return A
    
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class DescriptorEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([33] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    # def forward(self, kpts, scores):
    def forward(self, kpts):
        # 调换1,2维度
        # inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = [kpts.transpose(1, 2)]
        return self.encoder(torch.cat(inputs, dim=1))

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        # self.encoder = MLP([3] + layers + [feature_dim])
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
    # def forward(self, kpts):
        # 调换1,2维度;unsqueeze在第(1)维增加一个维度
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        # inputs = [kpts.transpose(1, 2)]
        return self.encoder(torch.cat(inputs, dim=1))

import torch.nn.functional as F

def Rot3DPositionEncoder(desc0, pos, lrf, resolving_distance):
    '''encode three dimensional position to rotation'''
    batch_dim, feature_dim, num_heads, num_points = desc0.size()
    _, _, pos_dim = pos.size()
    device=torch.device('cuda')

    # position = pos[:, :3, :]

    desc = desc0.view(batch_dim,np.int(feature_dim/2),2,num_heads,num_points)
    desc = torch.cat((-desc[:,:,1,:,:], desc[:,:,0,:,:]), 2)
    desc = desc.view(batch_dim,feature_dim,num_heads,num_points)

    # transform keypoint position 'pos' to Local Reference Frame
    # lrf: transform sensor frame to Local Reference Frame
    pos = torch.einsum('bdxy,bdy->bdx', lrf.transpose(2,3), pos)
    logpos = torch.log(torch.abs(pos)+1)
    logpos[pos<0] = -logpos[pos<0]
    logpos = F.interpolate(logpos, scale_factor=2, mode='nearest').unsqueeze(2)
    logpos = logpos.repeat(1, 1, num_heads, np.int(feature_dim/(2*pos_dim)))
    logpos = torch.transpose(logpos, 1, 3)

    theta = torch.arange(0, feature_dim/(2*pos_dim), device=device).view(1, 1, -1)
    theta = F.interpolate(theta, scale_factor=2*pos_dim, mode='nearest').view(1, -1, 1, 1)
    theta = theta.repeat(batch_dim, 1, num_heads, num_points)
    theta = 100**(-theta*2*pos_dim/feature_dim)
    theta = torch.mul(logpos, theta)

    return torch.mul(desc0,torch.cos(theta))+torch.mul(desc,torch.sin(theta))

def Rot3DRelPositionEncoder(desc0, pos, lrf):
    '''encode three dimensional position to rotation'''
    batch_dim, feature_dim, num_heads, num_points = desc0.size()
    _, _, m, _ = pos.size()
    pos = pos.float()
    lrf = lrf.float()
    desc0 = desc0.float()

    # position = pos[:, :3, :]
    desc = desc0.view(batch_dim,np.int(feature_dim/2),2,num_heads,num_points)
    desc = torch.cat((-desc[:,:,1,:,:], desc[:,:,0,:,:]), 2)
    desc = desc.view(batch_dim,feature_dim,num_heads,num_points)

    # transform keypoint position 'pos' to Local Reference Frame
    # transfer euclidean coordinate to polar coordinate
    pos = torch.einsum('bnxy,bnmy->bnmx', lrf.transpose(2,3), pos)
    alpha  = torch.arctan(pos[:,:,:,1]/pos[:,:,:,0])
    alpha[pos[:,:,:,0]<0] = alpha[pos[:,:,:,0]<0]+np.pi
    phi = torch.arctan(pos[:,:,:,2]/torch.sqrt(pos[:,:,:,0]**2 + pos[:,:,:,1]**2))
    # rho = torch.sqrt(pos[:,:,:,0]**2 + pos[:,:,:,1]**2 + pos[:,:,:,2]**2)
    # rho = rho.unsqueeze(1).repeat(1, num_heads, 1, 1)

    phi[torch.isnan(phi)] = 0
    alpha[torch.isnan(alpha)] = 0
    phi = phi.view(batch_dim, num_points, m)
    alpha = alpha.view(batch_dim, num_points, m)

    rot = torch.cat([alpha.unsqueeze(3), phi.unsqueeze(3)], 3)
    rot = F.interpolate(rot, size=[m, 4], mode='nearest').unsqueeze(2) # interpolate the dimension after 2th
    rot = rot.repeat(1, 1, num_heads, 1, np.int(feature_dim/(2*2)))

    value = torch.einsum('bdhn,bnhmd->bdhnm', desc0, torch.cos(rot)) + torch.einsum('bdhn,bnhmd->bdhnm', desc, torch.sin(rot))

    return value

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

# 通过注意力权重选择邻居
def dynamic_attention(query, key, value, k):
    batch, dim, head, n = query.shape
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    if k == None:
        # scores = torch.mul(scores, 1/(rho+1))
        scores = torch.nn.functional.softmax(scores, dim=-1)
    else:
        values = scores.topk(k, dim=3, largest=True, sorted=True).values[:,:,:,k-1] # the top k-th values
        values = values.unsqueeze(3).repeat(1,1,1,256)
        idx = scores<values
        # scores = torch.mul(scores, 1/(rho+1))
        prob = torch.nn.functional.softmax(scores[idx==False].view(batch, head, n, k), dim=-1) # perform softmax on the top k nodes
        scores[idx]=0
        scores[idx==False] = prob.view(-1)
        
    scores = scores.float()
    return torch.einsum('bhnm,bdhmn->bdhn', scores, value), scores


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, x, source, x_pos, src_pos, lrf, name, k):
        batch_dim, feature_dim, n = x.size()
        _, _, m = source.size()

        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (x, source, source))]
        rho=[]
        if name=='self':
            '''v1: position information augmented q,k,v'''
            # query = Rot3DPositionEncoder(query, x_pos, lrf, self.resolving_distance)
            # key = Rot3DPositionEncoder(key, src_pos, lrf, self.resolving_distance)
            # # value = Rot3DPositionEncoder(value, x_pos-src_pos, lrf, self.resolving_distance)
            # # value = Rot3DrelPositionEncoder(value, x_pos, src_pos, lrf, self.resolving_distance)

            '''v2: position augmentation using polar coordinates'''
            relative_pos = x_pos.unsqueeze(2).repeat(1,1,m,1) - src_pos.unsqueeze(1).repeat(1,n,1,1)
            # relative_pos = relative_pos[(relative_pos==0)==False].view(batch_dim, n,m-1,3)

            # rho = torch.sqrt(relative_pos[:,:,:,0]**2 + relative_pos[:,:,:,1]**2 + relative_pos[:,:,:,2]**2)
            # rho = rho.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            value = Rot3DRelPositionEncoder(value, relative_pos, lrf)
            x, prob = dynamic_attention(query, key, value, k)
            x = x.double()
        else:
            x, prob = attention(query, key, value)

        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, kpts0, kpts1, lrf, name, k):
        message = self.attn(x, source, kpts0, kpts1, lrf, name, k)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, kpts0, kpts1, lrf0, lrf1, k_list, L):
        i = 0
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
                src0_pos, src1_pos = kpts1, kpts0
            else:
                src0, src1 = desc0, desc1
                src0_pos, src1_pos = kpts0, kpts1                

            if i > 2*L-1-len(k_list):
                k = k_list[i-2*L+len(k_list)]
                delta0, delta1 = layer(desc0, src0, kpts0, src0_pos, lrf0, name, k), layer(desc1, src1, kpts1, src1_pos, lrf1, name, k)
            else:
                delta0, delta1 = layer(desc0, src0, kpts0, src0_pos, lrf0, name, None), layer(desc1, src1, kpts1, src1_pos, lrf1, name, None)
            
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            i+=1
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class r_MDGAT2(nn.Module):
    """
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In RAL-IROS, 2021.
    """
    default_config = {
        'descriptor_dim': 128,
        'keypoint_encoder': [32, 64, 128],
        'descritor_encoder': [64, 128],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'var_weight': 1
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.descriptor = config['descriptor']
        if self.descriptor == 'FPFH':
            self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
            
            self.denc = DescriptorEncoder(
            self.config['descriptor_dim'], self.config['descritor_encoder'])

        elif self.descriptor == 'FPFH_only':
            self.denc = DescriptorEncoder(
            self.config['descriptor_dim'], self.config['descritor_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], ['self', 'cross']*self.config['L'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.lr = config['lr']
        self.loss_method = config['loss_method']
        self.k = config['k']
        self.mutual_check = config['mutual_check']
        self.triplet_loss_gamma = config['triplet_loss_gamma']
        self.local_rank = config['local_rank']

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        
        kpts0, kpts1 = data['keypoints0'].double(), data['keypoints1'].double()
        lrf0, lrf1 = data['lrf0'].double(), data['lrf1'].double()
    
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }
        # file_name = data['file_name']
        
        # Keypoint normalization.
        # kpts0 = normalize_keypoints(kpts0, data['cloud0'].shape)
        # kpts1 = normalize_keypoints(kpts1, data['cloud1'].shape)

        if self.descriptor == 'FPFH':
            desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
            # Keypoint MLP encoder.
            desc0 = self.denc(desc0) + self.kenc(kpts0, data['scores0'])
            desc1 = self.denc(desc1) + self.kenc(kpts1, data['scores1'])
            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1, kpts0, kpts1, lrf0, lrf1, self.k, self.config['L'])
            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        else:
            raise Exception('Invalid descriptor.')

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        gt_matches0 = data['match0'] # shape=torch.Size([1, 87, 2])
        gt_matches1 = data['match1'] # shape=torch.Size([1, 87, 2])

        # 输出match结果
        if self.loss_method == 'superglue':
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            zero = scores.new_tensor(0)
            if self.mutual_check:
                mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0) #gather 沿给定轴dim，将输入索引张量index指定位置的值进行聚合。
                mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
                mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
                valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
                valid1 = mutual1 & valid0.gather(1, indices1)
            else:
                valid0 = max0.values.exp() > self.config['match_threshold']
                valid1 = max1.values.exp() > self.config['match_threshold']
                mscores0 = torch.where(valid0, max0.values.exp(), zero)
                mscores1 = torch.where(valid1, max1.values.exp(), zero)
        else:
            max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            valid0, valid1 = indices0<(scores.size(2)-1), indices1<(scores.size(1)-1)
            zero = scores.new_tensor(0)
            if valid0.sum() == 0:
                mscores0 = torch.zeros_like(indices0, device='cuda')
                mscores1 = torch.zeros_like(indices1, device='cuda')
            else:
                if self.mutual_check:
                    batch = indices0.size(0)
                    a0 = arange_like(indices0, 1)[None][valid0].view(batch,-1) == indices1.gather(1, indices0[valid0].view(batch,-1))
                    a1 = arange_like(indices1, 1)[None][valid1].view(batch,-1) == indices0.gather(1, indices1[valid1].view(batch,-1))
                    mutual0 = torch.zeros_like(indices0, device='cuda') > 0
                    mutual1 = torch.zeros_like(indices1, device='cuda') > 0
                    mutual0[valid0] = a0
                    mutual1[valid1] = a1
                    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                    mscores1 = torch.where(mutual1, max1.values.exp(), zero)
                else:
                    mscores0 = torch.where(valid0, max0.values.exp(), zero)
                    mscores1 = torch.where(valid1, max1.values.exp(), zero)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        
        # calculate loss
        if self.loss_method == 'superglue':
            loss = superglue()
        elif self.loss_method == 'triplet_loss':
            loss = triplet(self.triplet_loss_gamma)
        elif self.loss_method == 'gap_loss':
            loss = gap(self.triplet_loss_gamma)
        elif self.loss_method == 'gap_loss_plus':
            loss_mean = gap_plus(gt_matches0, gt_matches1, scores, self.triplet_loss_gamma, self.config['var_weight'])
        elif self.loss_method == 'gap_loss_plusplus':
            loss_mean = gap_plusplus(gt_matches0, gt_matches1, scores, self.triplet_loss_gamma, self.config['var_weight'])
        elif self.loss_method == 'distribution_loss':
            loss = distribution(self.triplet_loss_gamma)
        elif self.loss_method == 'distribution_loss6':
            loss = distribution6(self.triplet_loss_gamma, self.lamda)
        elif self.loss_method == 'distribution_loss7':
            loss = distribution7(self.triplet_loss_gamma, self.lamda)
        
        if torch.cuda.is_available():
            device=torch.device('cuda:{}'.format(self.local_rank[0]))
            if torch.cuda.device_count() > 1:
                loss = torch.nn.DataParallel(loss, device_ids=self.local_rank)
            else:
                loss = torch.nn.DataParallel(loss)
        else:
            device = torch.device("cpu")
        loss.to(device)

        if self.loss_method == 'distribution_loss':
            loss_mean = loss(gt_matches0, gt_matches1, scores)
        else:
            b,d,n = mdesc0.size()
            _,_,m = mdesc1.size()
            distance = mdesc0[:,:,:,None].expand(b,d,n,m) - mdesc1[:,:,None].expand(b,d,n,m)
            distance = torch.sqrt(torch.sum(distance**2, 1)/d)
            loss_mean = loss(gt_matches0, gt_matches1, scores, distance)

        # loss_mean = loss(gt_matches0, gt_matches1, scores)
            
        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'loss': loss_mean,
            # 'skip_train': False
        }

    def update_lamda(self, epoch, indicator):
        lamda_clip = 1
        self.lamda = 0.02*(epoch-indicator) + self.lamda_initial

        if self.lamda > lamda_clip:
            self.lamda = lamda_clip
