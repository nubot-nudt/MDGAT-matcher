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
import torch
from torch import nn
import time

from models.pointnet.pointnet_util import PointNetSetKptsMsg, PointNetSetAbstraction


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

        # self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        # self.kenc = KeypointEncoder(feature_dim, layers)

    def forward(self, xyz, kpts, score):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm, kpts)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        desc = l2_points.view(B, 128, -1)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.fc3(x)
        # desc = F.log_softmax(desc, -1)

        # kpts = self.kenc(kpts, score)
        # desc = self.mlp(torch.cat([kpts, desc], dim=1))

        return desc

class PointnetEncoder(nn.Module):
    def __init__(self,feature_dim: int,layers,normal_channel=True):
        super().__init__()
        in_channel = 5 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetKptsMsg(256, [1], [32], in_channel, [[64, 64, 128]])
        self.sa2 = PointNetSetAbstraction(None, None, None, 128 + 3, [256, 256, 128], True)
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, feature_dim)

        # self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        # self.kenc = KeypointEncoder(feature_dim, layers)

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

        # kpts = self.kenc(kpts, score)
        # print('kenc',time.time() - begin)
        # begin = time.time()
        # desc = self.mlp(torch.cat([kpts, desc], dim=1))
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
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

class DescriptorEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([33] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
        # self.encoder2 = MLP([feature_dim*2, feature_dim*2, feature_dim])
        # nn.init.constant_(self.encoder2[-1].bias, 0.0)

    # def forward(self, kpts, scores):
    def forward(self, kpts):
        inputs = [kpts.transpose(1, 2)]
        desc = self.encoder(torch.cat(inputs, dim=1))

        return desc

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
        # inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        inputs = [kpts.transpose(1, 2)]
        desc = self.encoder(torch.cat(inputs, dim=1))
        b, dim, n = desc.size()
        gloab = torch.max(desc, dim=2)[0].view(b, dim, 1)
        gloab = gloab.repeat(1,1,n)
        desc = torch.cat([desc, gloab], 1)
        desc = self.encoder2(desc)
        return desc

# global aware
class pointnetDescriptorEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, desc):
        b, dim, n = desc.size()
        gloab = torch.max(desc, dim=2)[0].view(b, dim, 1)
        gloab = gloab.repeat(1,1,n)
        desc = torch.cat([desc, gloab], 1)
        desc = self.encoder(desc)
        return desc

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
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

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
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


class SuperGlue(nn.Module):
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
        'keypoint_encoder': [32, 64, 128],
        'descritor_encoder': [64, 128],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.descriptor = config['descriptor']
        if self.descriptor == 'pointnet':
            self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
            self.denc = pointnetDescriptorEncoder(
            self.config['descriptor_dim'], self.config['descritor_encoder'])

            self.penc = PointnetEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        elif self.descriptor == 'pointnetmsg':
            self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
            self.denc = pointnetDescriptorEncoder(
            self.config['descriptor_dim'], self.config['descritor_encoder'])

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
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.lr = config['lr']
        self.loss_method = config['loss_method']
        self.mutual_check = config['mutual_check']
        self.triplet_loss_gamma = config['triplet_loss_gamma']
        self.train_step = config['train_step']


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
            
        if self.descriptor == 'FPFH' or self.descriptor == 'FPFH_gloabal':
            desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
            '''Keypoint MLP encoder.'''
            desc0 = self.denc(desc0) + self.kenc(kpts0, data['scores0'])
            desc1 = self.denc(desc1) + self.kenc(kpts1, data['scores1'])
            '''Multi-layer Transformer network.'''
            desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
            '''Final MLP projection.'''
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        elif self.descriptor == 'pointnet' or self.descriptor == 'pointnetmsg':
            pc0, pc1 = data['cloud0'].double(), data['cloud1'].double()
            desc0 = self.penc(pc0, kpts0, data['scores0'])
            desc1 = self.penc(pc1, kpts1, data['scores1'])
            """
            3-step training
            """
            
            if self.train_step == 1: # update pointnet only
                mdesc0, mdesc1 = desc0, desc1 
            elif self.train_step == 2: # update gnn only      
                desc0, desc1 = desc0.detach(), desc1.detach()
                '''Multi-layer Transformer network.'''
                desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
                '''Final MLP projection.'''
                mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
            elif self.train_step == 3: # update pointnet and gnn
                '''Multi-layer Transformer network.'''
                desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
                '''Final MLP projection.'''
                mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
            else:
                raise Exception('Invalid train_step.')
        elif self.descriptor == 'FPFH_only':
            desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
            desc0 = self.denc(desc0) 
            desc1 = self.denc(desc1) 
            desc0, desc1 = self.gnn(desc0, desc1, self.k, self.config['L'])
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        else:
            raise Exception('Invalid descriptor.')

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        '''Run the optimal transport.'''
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        gt_matches0 = data['match0']
        gt_matches1 = data['match1']

        '''Match the keypoints'''
        if self.loss_method == 'superglue':
            '''determine the non_matches by comparing to a pre-defined threshold'''
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            zero = scores.new_tensor(0)
            if self.mutual_check:
                mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0) 
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
            '''directly determine the non_matches on scores matrix'''
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
            neg_indice[max0[:,:,0] == gt_matches0] = 1 
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
        elif self.loss_method == 'gap_loss':
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
        }
