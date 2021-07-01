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

def knn(x, src, k):
    inner = -2*torch.matmul(x.transpose(2, 1), src)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    ss = torch.sum(src**2, dim=1, keepdim=True)
    pairwise_distance = -xx.transpose(2, 1) - inner - ss
    # 距离取负，所以topk就是距离最小的k个。[0]返回值，[1]返回index
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, src, k, idx=None):
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


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    # _, _, height, width = image_shape
    _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

import torch.nn.functional as F
class PointnetEncoderMsg(nn.Module):
    def __init__(self,feature_dim: int,layers,normal_channel=True):
        super().__init__()
        in_channel = 5 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetKptsMsg(256, [1, 1.5, 2.25], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa2 = PointNetSetAbstraction(None, None, None, 320 + 3, [256, 256, 128], True)
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, num_class)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        self.kenc = KeypointEncoder(feature_dim, layers)

    def forward(self, xyz, kpts, score):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # begin = time.time()
        l1_xyz, l1_points = self.sa1(xyz, norm, kpts)
        # print('sa1',time.time() - begin)
        # begin = time.time()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print('sa2',time.time() - begin)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        desc = l2_points.view(B, 128, -1)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # desc = F.log_softmax(desc, -1)
        # begin = time.time()
        kpts = self.kenc(kpts, score)
        # print('kenc',time.time() - begin)
        # begin = time.time()
        desc = self.mlp(torch.cat([kpts, desc], dim=1))
        # print('mlp',time.time() - begin)
        return desc

class PointnetEncoder(nn.Module):
    def __init__(self,feature_dim: int,layers,normal_channel=True):
        super().__init__()
        in_channel = 5 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetKptsMsg(256, [2], [32], in_channel, [[64, 64, 128]])
        self.sa2 = PointNetSetAbstraction(None, None, None, 128 + 3, [256, 256, 128], True)
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        self.kenc = KeypointEncoder(feature_dim, layers)

    def forward(self, xyz, kpts, score):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # begin = time.time()
        l1_xyz, l1_points = self.sa1(xyz, norm, kpts)
        # print('sa1',time.time() - begin)
        # begin = time.time()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print('sa2',time.time() - begin)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        desc = l2_points.view(B, 128, -1)
        # desc = self.drop1(F.relu(self.bn1(self.fc1(desc))))
        # desc = self.drop2(F.relu(self.bn2(self.fc2(desc))))
        # desc = self.fc3(desc)
        # desc = F.log_softmax(desc, -1)
        # begin = time.time()
        kpts = self.kenc(kpts, score)
        # print('kenc',time.time() - begin)
        # begin = time.time()
        desc = self.mlp(torch.cat([kpts, desc], dim=1))
        # print('mlp',time.time() - begin)
        return desc

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

#sch
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

class DescriptorGloabalEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([33] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
        self.encoder2 = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.encoder2[-1].bias, 0.0)

    # def forward(self, kpts, scores):
    def forward(self, kpts):
        # 调换1,2维度
        # inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = [kpts.transpose(1, 2)]
        desc = self.encoder(torch.cat(inputs, dim=1))
        b, dim, n = desc.size()
        gloab = torch.max(desc, dim=2)[0].view(b, dim, 1)
        gloab = gloab.repeat(1,1,n)
        desc = torch.cat([desc, gloab], 1)
        desc = self.encoder2(desc)
        return desc


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

# 只针对self attention
def dynamic_attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhnk->bhnk', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnk,bdhnk->bdhn', prob, value), prob

# 通过比较输入向量的相似性选择邻域
def dynamic_attention2(query, key, value, A, k):
    batch, dim, head, n = query.shape
    m = key.shape[3]
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    scores = torch.einsum('bhnm,bnm->bhnm', scores, A) 
    # value = value[A.bool()].view(batch, dim, head, n)
    # 直接softmax，0权重处理过后也会有值
    A = A.view(batch, 1, n, m).repeat(1, head, 1, 1).bool()
    scores = scores[A].view(batch, head, n, k)
    S = torch.nn.functional.softmax(scores, dim=-1)
    prob = torch.zeros([batch, head, n, m], dtype=float, device=torch.device('cuda'))
    prob[A] = S.view(-1)

    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

# 通过注意力权重选择邻居
def dynamic_attention3(query, key, value, k):
    batch, dim, head, n = query.shape
    m = key.shape[3]
    dim = query.shape[1]
    device=torch.device('cuda')
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    K = scores.topk(k, dim=3, largest=True, sorted=True).indices
    B = torch.arange(0, batch, device=device).view(-1, 1, 1, 1).repeat(1, head, n, k)
    H = torch.arange(0, head, device=device).view(1, -1, 1, 1).repeat(batch, 1, n, k)
    N = torch.arange(0, n, device=device).view(1, 1, -1, 1).repeat(batch, head, 1, k)
    scores = scores[B,H,N,K]
    S = torch.nn.functional.softmax(scores, dim=-1)
    prob = torch.zeros([batch, head, n, m], dtype=float, device=device)
    prob[B,H,N,K] = S
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    # def forward(self, x, source, name, k):
    def forward(self, x, source, k):
        batch_dim, feature_dim, num_points = x.size()
        
        # if name == 'cross':
        #     query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                         for l, x in zip(self.proj, (x, source, source))]
        #     # x, prob = attention(query, key, value)
        #     x, prob = attention(query, key, value)
        # else:
        #     A = get_graph_feature(x, source, k)
        #     query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                         for l, x in zip(self.proj, (x, source, source))]
        #     # A = A.view(batch_dim, 1, num_points, num_points).repeat(1, self.num_heads, 1, 1)
        #     x, prob = dynamic_attention2(query, key, value, A, k)
    
        if k==None:
            query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (x, source, source))]
            # key = key.view(batch_dim, self.dim, self.num_heads, -1, k)#bdhnk
            # value = value.view(batch_dim, self.dim, self.num_heads, -1, k)
            x, prob = attention(query, key, value)
        else:
            """ 根据比较输入向量的相似性选择邻居 """
            # A = get_graph_feature(x, source, k)
            # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
            #                     for l, x in zip(self.proj, (x, source, source))]
            # # A = A.view(batch_dim, 1, num_points, num_points).repeat(1, self.num_heads, 1, 1)
            # x, prob = dynamic_attention2(query, key, value, A, k)

            """ 通过注意力权重选择邻居 """
            query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (x, source, source))]
            x, prob = dynamic_attention3(query, key, value, k)

            #TEST
            # source2, A = get_graph_feature(x, source, k)
            # source2 = source2.view(batch_dim, self.dim, self.num_heads, -1, k)#bdhnk
            # x = x.view(batch_dim, self.dim, self.num_heads, -1)
            # source = source.view(batch_dim, self.dim, self.num_heads, -1)
            # # A = A.view(batch_dim, 1, num_points, num_points).repeat(1, self.num_heads, 1, 1)
           
            # x2, prob = dynamic_attention(x, source2, source2)
            # x3, prob2 = dynamic_attention2(x, source, source, A, k)
            
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    # def forward(self, x, source, name, k):
    #     message = self.attn(x, source, name, k)
    def forward(self, x, source, k):
        message = self.attn(x, source, k)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, k_list, L):
        i = 0
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1

            if i > 2*L-1-len(k_list):
                k = k_list[i-2*L+len(k_list)]
                delta0, delta1 = layer(desc0, src0, k), layer(desc1, src1, k)
            else:
                delta0, delta1 = layer(desc0, src0, None), layer(desc1, src1, None)

            # delta0, delta1 = layer(desc0, src0, name, k_list), layer(desc1, src1, name, k_list)
            
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


class MDGAT(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 128,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128],
        'descritor_encoder': [64, 128],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.descriptor = config['descriptor']
        if self.descriptor == 'pointnet':
            self.penc = PointnetEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        elif self.descriptor == 'pointnetmsg':
            self.penc = PointnetEncoderMsg(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        elif self.descriptor == 'FPFH':
            self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
            
            self.denc = DescriptorEncoder(
            self.config['descriptor_dim'], self.config['descritor_encoder'])
        elif self.descriptor == 'FPFH_gloabal':
            self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
            
            self.denc = DescriptorGloabalEncoder(
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
        self.train_step = config['train_step']

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        
        kpts0, kpts1 = data['keypoints0'].double(), data['keypoints1'].double()
    
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

        if self.descriptor == 'FPFH' or self.descriptor == 'FPFH_gloabal':
            desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
            # Keypoint MLP encoder.
            desc0 = self.denc(desc0) + self.kenc(kpts0, data['scores0'])
            desc1 = self.denc(desc1) + self.kenc(kpts1, data['scores1'])
            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        elif self.descriptor == 'pointnet' or self.descriptor == 'pointnetmsg':
            # begin = time.time()
            pc0, pc1 = data['cloud0'].double(), data['cloud1'].double()
            desc0 = self.penc(pc0, kpts0, data['scores0'])
            desc1 = self.penc(pc1, kpts1, data['scores1'])
            # print('encoder',time.time() - begin)
            """
            3-step训练
            """
            # 只更新pointnet
            if self.train_step == 1:
                mdesc0, mdesc1 = desc0, desc1
            # 不更新pointnet
            elif self.train_step == 2:                 
                desc0, desc1 = desc0.detach(), desc1.detach()
                # Multi-layer Transformer network.
                desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
                # Final MLP projection.
                mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
            # 更新pointnet和gnn
            elif self.train_step == 3: 
                # Multi-layer Transformer network.
                desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
                # Final MLP projection.
                mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
            else:
                raise Exception('Invalid train_step.')
        elif self.descriptor == 'FPFH_only':
            desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()

            # x= self.kenc(kpts0, data['scores0'])
            desc0 = self.denc(desc0) 
            desc1 = self.denc(desc1) 
            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
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
        if self.loss_method == 'triplet_loss' or self.loss_method == 'gap_loss3':
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
        else:
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
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        

        if self.loss_method == 'superglue':
            aa = time.time()
            b, n = gt_matches0.size()
            _, m = gt_matches1.size()
            batch_idx = torch.arange(0, b, dtype=int, device=torch.device('cuda')).view(-1, 1).repeat(1, n)
            idx = torch.arange(0, n, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
            indice = gt_matches0.long()
            loss_tp = scores[batch_idx, idx, indice]
            loss_tp = torch.sum(loss_tp, dim=1)

            indice = gt_matches1.long()
            mutual_inv = torch.zeros_like(gt_matches1, dtype=int, device=torch.device('cuda')) > 0
            mutual_inv[gt_matches1 == -1] = True
            xx = mutual_inv.sum(1)
            loss_all = scores[batch_idx[mutual_inv], indice[mutual_inv], idx[mutual_inv]]
            for batch in range(len(gt_matches0)):
                xx_sum = torch.sum(xx[:batch])
                loss = loss_all[xx_sum:][:xx[batch]]
                loss = torch.sum(loss).view(1)
                if batch == 0:
                    loss_tn = loss
                else:
                    loss_tn = torch.cat((loss_tn, loss))
            loss_mean = torch.mean((-loss_tp - loss_tn)/(xx+m))
            loss_all = torch.mean(torch.cat((loss_tp.view(-1), loss_all)))

            # bb = time.time()
            # loss_mean2 = 0
            # loss22 =[]
            # loss33 =[]
            # for batch1 in range(len(gt_matches0)):
            #     loss2 =[]
            #     loss3 =[]
            #     for i in range(len(gt_matches0[batch1])):
            #         x = i
            #         y = gt_matches0[batch1][i]
            #         # 匹配点的loss
            #         loss2.append(-torch.log(scores[batch1][x][y].exp())) # check batch size == 1 ?
            #     # matches1中没有统计的无匹配点
            #     loss22.append(torch.sum(torch.stack(loss2)))                                                                                                    
            #     for i in range(len(gt_matches1[batch1])):
            #         x = gt_matches1[batch1][i]
            #         y = i
            #         if x == -1:
            #             loss2.append(-torch.log(scores[batch1][x][y].exp())) 
            #             loss3.append(-torch.log(scores[batch1][x][y].exp()))
            #     loss_mean2 += torch.mean(torch.stack(loss2))
            # # loss_mean = torch.mean(torch.stack(loss2))
            #     # loss22.append(torch.sum(torch.stack(loss2)))
            #     loss33.append(torch.sum(torch.stack(loss3)))
            # loss_mean2 = loss_mean2/b
            # cc = time.time()
            # print(bb-aa,' ',cc-bb)
            # print()
        elif self.loss_method == 'mine':
            loss_tn = []
            loss_tp = []
            loss_mean = 0
            
            for b in range(len(gt_matches0)):
                for i in range(len(gt_matches0[b])):
                    x = i
                    y = gt_matches0[b][i]
                    if y == -1:
                        loss_tn.append(-torch.log(scores[b][x][y].exp()))
                    else:
                        loss_tp.append(-torch.log(scores[b][x][y].exp())) # check batch size == 1 ?
                # matches1中没有统计的无匹配点                                                                                                    
                for i in range(len(gt_matches1[b])):
                    x = gt_matches1[b][i]
                    y = i
                    if x == -1:
                        loss_tn.append(-torch.log(scores[b][x][y].exp()))  
                    else:
                        loss_tp.append(-torch.log(scores[b][x][y].exp())) # check batch size == 1 ?
                # loss_mean += (torch.mean(torch.stack(loss_tn)) + torch.mean(torch.stack(loss_tp)))/2
            # @todo: 动态的loss比重 
            # loss_mean = loss_mean/len(gt_matches0)
            loss_mean = (torch.mean(torch.stack(loss_tn)) + torch.mean(torch.stack(loss_tp)))/2
        elif self.loss_method == 'triplet_loss':
            b, n = gt_matches0.size()
            _, m = gt_matches1.size()
            
            aa = time.time()
            max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
            max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
            gt_matches0[gt_matches0 == -1] = m
            gt_matches1[gt_matches1 == -1] = n
            batch_idx = torch.arange(0, b, dtype=int, device=torch.device('cuda')).view(-1, 1).repeat(1, n)
            idx = torch.arange(0, n, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)

            # pc0 -> pc1   
            pos_indice = gt_matches0.long()
            neg_indice = torch.zeros_like(gt_matches0, dtype=int, device=torch.device('cuda'))
            neg_indice[max0[:,:,0] == gt_matches0] = 1 # 如果最大值就是真值，那么neg就选择次大值；否则neg选择最大值
            accuracy = neg_indice.sum().item()/neg_indice.size(1)
            neg_indice = max0[batch_idx, idx, neg_indice]
            loss_anc_neg = scores[batch_idx, idx, neg_indice]
            loss_anc_pos = scores[batch_idx, idx, pos_indice]
            

            # pc1 -> pc0
            pos_indice = gt_matches1.long()
            neg_indice = torch.zeros_like(gt_matches1, dtype=int, device=torch.device('cuda'))
            neg_indice[max1[:,0,:] == gt_matches1] = 1
            neg_indice = max1[batch_idx, neg_indice, idx]
            loss_anc_neg = torch.cat((loss_anc_neg, scores[batch_idx, neg_indice, idx]), dim=1)
            loss_anc_pos = torch.cat((loss_anc_pos, scores[batch_idx, pos_indice, idx]), dim=1)

            loss_anc_neg = -torch.log(loss_anc_neg.exp())
            loss_anc_pos = -torch.log(loss_anc_pos.exp())
            before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
            active_percentage = torch.mean((before_clamp_loss > 0).float(), dim=1, keepdim=False)
            triplet_loss = torch.clamp(before_clamp_loss, min=0)
            loss_mean = torch.mean(triplet_loss)
            
            # gt_matches0[gt_matches0 == m] = -1
            # gt_matches1[gt_matches1 == n] = -1
            # bb=time.time()

            # loss_anc_pos1 = []
            # loss_anc_neg1 = []
            # for b in range(len(gt_matches0)):
            #     for i in range(len(gt_matches0[b])):
            #         x = i
            #         y = gt_matches0[b][i]
            #         loss_anc_pos1.append(-torch.log(scores[b][x][y].exp()))

            #         if y == -1:
            #             y = len(scores[b])-1
            #         if max0[b][x][0] == y:
            #             loss_anc_neg1.append(-torch.log(scores[b][x][max0[b][x][1]].exp()))
            #         else:
            #             loss_anc_neg1.append(-torch.log(scores[b][x][max0[b][x][0]].exp()))

            #     # matches1中没有统计的无匹配点                                                                                                    
            #     for i in range(len(gt_matches1[b])):
            #         x = gt_matches1[b][i]
            #         y = i
            #         loss_anc_pos1.append(-torch.log(scores[b][x][y].exp()))
            #         if x == -1:
            #             x = len(scores[b])-1
            #         if max1[b][0][y] == x:
            #             loss_anc_neg1.append(-torch.log(scores[b][max1[b][1][y]][y].exp()))
            #         else:
            #             loss_anc_neg1.append(-torch.log(scores[b][max1[b][0][y]][y].exp()))

                # loss_mean += loss_anc_pos1 - loss_anc_neg1 + self.triple_loss_gamma
            # @todo: 动态的loss比重 
            # loss_mean = loss_mean/len(gt_matches0)
            # loss_mean = (torch.mean(torch.stack(loss_tn)) + torch.mean(torch.stack(loss_tp)))/2
            # loss_mean = torch.reshape(loss_mean, (1, -1))
            # cc=time.time()
            # print(bb-aa,' ',cc-bb)
            # print(' ')
        elif self.loss_method == 'gap_loss':
            b, n = gt_matches0.size()
            _, m = gt_matches1.size()
            
            aa = time.time()
            max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
            max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
            gt_matches0[gt_matches0 == -1] = m
            gt_matches1[gt_matches1 == -1] = n
            batch_idx = torch.arange(0, b, dtype=int, device=torch.device('cuda')).view(-1, 1).repeat(1, n)
            idx = torch.arange(0, n, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)

            # pc0 -> pc1   
            pos_indice = gt_matches0.long()
            neg_indice = torch.zeros_like(gt_matches0, dtype=int, device=torch.device('cuda'))
            neg_indice[max0[:,:,0] == gt_matches0] = 1 # 如果最大值就是真值，那么neg就选择次大值；否则neg选择最大值
            accuracy = neg_indice.sum().item()/neg_indice.size(1)
            neg_indice = max0[batch_idx, idx, neg_indice]
            loss_anc_neg = scores[batch_idx, idx, neg_indice]
            loss_anc_pos = scores[batch_idx, idx, pos_indice]
            

            # pc1 -> pc0
            pos_indice = gt_matches1.long()
            neg_indice = torch.zeros_like(gt_matches1, dtype=int, device=torch.device('cuda'))
            neg_indice[max1[:,0,:] == gt_matches1] = 1
            neg_indice = max1[batch_idx, neg_indice, idx]
            loss_anc_neg = torch.cat((loss_anc_neg, scores[batch_idx, neg_indice, idx]), dim=1)
            loss_anc_pos = torch.cat((loss_anc_pos, scores[batch_idx, pos_indice, idx]), dim=1)

            loss_anc_neg = -torch.log(loss_anc_neg.exp())
            loss_anc_pos = -torch.log(loss_anc_pos.exp())
            before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
            active_percentage = torch.mean((before_clamp_loss > 0).float(), dim=1, keepdim=False)
            triplet_loss = torch.mul(torch.clamp(before_clamp_loss, min=0),loss_anc_pos)
            loss_mean = torch.mean(triplet_loss)
        elif self.loss_method == 'gap_loss2':
            b, n = gt_matches0.size()
            _, m = gt_matches1.size()
            
            aa = time.time()
            # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
            # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
            gt_matches0[gt_matches0 == -1] = m
            gt_matches1[gt_matches1 == -1] = n
            
            idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
            idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
            idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
            # pc0 -> pc1   
            pos_indice = gt_matches0.long()
            # determine neg indice
            pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
            pos_match = idx == pos_indice2
            neg_match = pos_match == False
            loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
            loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
            loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
            loss_anc_neg = -torch.log(loss_anc_neg.exp())
            loss_anc_pos = -torch.log(loss_anc_pos.exp())
            before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
            active_num = torch.sum((before_clamp_loss > 0).float(), dim=2, keepdim=False)
            active_num = torch.clamp(active_num-1, min=0)
            active_num = torch.sum(active_num, dim=1)
            gap_loss = torch.clamp(before_clamp_loss, min=0)
            loss_sum = torch.sum(torch.sum(gap_loss, dim=2), dim=1)
            loss_mean = torch.mul(loss_sum, 1/(active_num+n))

            # pc1 -> pc0
            pos_indice = gt_matches1.long()
            # determine neg indice
            pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
            pos_match = idx2 == pos_indice2
            neg_match = pos_match == False
            loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
            loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
            loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
            loss_anc_neg = -torch.log(loss_anc_neg.exp())
            loss_anc_pos = -torch.log(loss_anc_pos.exp())
            before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
            active_num = torch.sum((before_clamp_loss > 0).float(), dim=1, keepdim=False)
            active_num = torch.clamp(active_num-1, min=0)
            active_num = torch.sum(active_num, dim=1)
            gap_loss = torch.clamp(before_clamp_loss, min=0)
            loss_sum = torch.sum(torch.sum(gap_loss, dim=1), dim=1)
            loss_mean2 = torch.mul(loss_sum, 1/(active_num+m))

            loss_mean = (loss_mean+loss_mean2)/2
        elif self.loss_method == 'gap_loss3':
            b, n = gt_matches0.size()
            _, m = gt_matches1.size()
            
            aa = time.time()
            # max0 = scores[:,:-1,:].topk(2, dim=2, largest=True, sorted=True).indices
            # max1 = scores[:,:,:-1].topk(2, dim=1, largest=True, sorted=True).indices
            gt_matches0[gt_matches0 == -1] = m
            gt_matches1[gt_matches1 == -1] = n
            
            idx = torch.arange(0, m+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, n, 1)
            idx2 = torch.arange(0, n+1, dtype=int, device=torch.device('cuda')).view(1, -1).repeat(b, 1)
            idx2 = torch.repeat_interleave(idx2.unsqueeze(dim=2),repeats=m,dim=2)
            # pc0 -> pc1   
            pos_indice = gt_matches0.long()
            # determine neg indice
            pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=2),repeats=m+1,dim=2)
            pos_match = idx == pos_indice2
            neg_match = pos_match == False
            loss_anc_pos = scores[:,:-1,:][pos_match].view(b,n)
            loss_anc_neg = scores[:,:-1,:][neg_match].view(b,n,m)
            loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=2),repeats=m,dim=2)
            loss_anc_neg = -torch.log(loss_anc_neg.exp())
            loss_anc_pos = -torch.log(loss_anc_pos.exp())
            before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
            active_num = torch.sum((before_clamp_loss > 0).float(), dim=2, keepdim=False)
            gap_loss = torch.clamp(before_clamp_loss, min=0)
            loss_mean = torch.mean(2*torch.log(torch.sum(gap_loss, dim=2)+1), dim=1)

            # pc1 -> pc0
            pos_indice = gt_matches1.long()
            # determine neg indice
            pos_indice2 = torch.repeat_interleave(pos_indice.unsqueeze(dim=1),repeats=n+1,dim=1)
            pos_match = idx2 == pos_indice2
            neg_match = pos_match == False
            loss_anc_pos = scores[:,:,:-1][pos_match].view(b,m)
            loss_anc_neg = scores[:,:,:-1][neg_match].view(b,n,m)
            loss_anc_pos = torch.repeat_interleave(loss_anc_pos.unsqueeze(dim=1),repeats=n,dim=1)
            loss_anc_neg = -torch.log(loss_anc_neg.exp())
            loss_anc_pos = -torch.log(loss_anc_pos.exp())
            before_clamp_loss = loss_anc_pos - loss_anc_neg + self.triplet_loss_gamma
            active_num = torch.sum((before_clamp_loss > 0).float(), dim=1, keepdim=False)
            active_num = torch.clamp(active_num-1, min=0)
            active_num = torch.sum(active_num, dim=1)
            gap_loss = torch.clamp(before_clamp_loss, min=0)
            loss_mean2 = torch.mean(2*torch.log(torch.sum(gap_loss, dim=1)+1), dim=1)

            loss_mean = (loss_mean+loss_mean2)/2

            
        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'loss': loss_mean,
            # 'skip_train': False
        }

        # scores big value or small value means confidence? log can't take neg value

    def update_learning_rate(self, ratio, optimizer):
        lr_clip = 0.00005

        # detector
        lr_detector = self.lr * ratio
        if lr_detector < lr_clip:
            lr_detector = lr_clip
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_detector
        # print('update detector learning rate: %f -> %f' % (self.lr, lr_detector))
        self.lr = lr_detector