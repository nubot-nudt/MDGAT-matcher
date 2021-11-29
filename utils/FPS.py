from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import torch.multiprocessing
from tqdm import tqdm
import time

import open3d as o3d
import pykitti
# visualize
import torchvision
from torchvision import transforms
# from logger import Logger
from tensorboardX import SummaryWriter

from models.fa.superglue import SuperGlue
from models.fa.r_mdgat import r_MDGAT
from models.fa.r_mdgat2 import r_MDGAT2
from models.fa.r_mdgat3 import r_MDGAT3
from models.fa.r_mdgat4 import r_MDGAT4
from models.fa.mdgat import MDGAT

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


from torch.utils.data import Dataset




class SparseDataset(Dataset):
    """Sparse correspondences dataset.  
    Reads images from files and creates pairs. It generates keypoints, 
    descriptors and ground truth matches which will be used in training."""

    def __init__(self, opt):
        
        self.train_path = opt.train_path

        self.txt_path = opt.txt_path
        self.dataset, self.seq_list = self.make_dataset_kitti_distance(self.txt_path)
        self.remove_outlier = opt.remove_outlier
        self.save_remove_outlier_file = opt.save_remove_outlier_file


    def __len__(self):
        return len(self.dataset)

    def load_kitti_gt_txt(self, txt_root, seq):
        '''
        :param txt_root:
        :param seq
        :return: [{anc_idx: *, pos_idx: *, seq: *}]                
        '''
        dataset = []
        with open(os.path.join(txt_root, '%02d'%seq, 'groundtruths.txt'), 'r') as f:
            lines_list = f.readlines()
            for i, line_str in enumerate(lines_list):
                if i == 0:
                    # skip the header line
                    continue
                line_splitted = line_str.split()
                anc_idx = int(line_splitted[0])
                pos_idx = int(line_splitted[1])
                # trans = [float(line_splitted[2]),float(line_splitted[3]),float(line_splitted[4])]
                # rot = [float(line_splitted[5]),float(line_splitted[6]),float(line_splitted[7]),float(line_splitted[8])]

                # # search for existence
                # anc_idx_is_exist = False
                # pos_idx_is_exist = False
                # for tmp_data in dataset:
                #     if tmp_data['anc_idx'] == anc_idx:
                #         anc_idx_is_exist = True
                #     if tmp_data['anc_idx'] == pos_idx:
                #         pos_idx_is_exist = True

                # if anc_idx_is_exist is False:
                #     data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
                #     dataset.append(data)
                # if pos_idx_is_exist is False:
                #     data = {'seq': seq, 'anc_idx': pos_idx, 'pos_idx': anc_idx}
                    # dataset.append(data)

                # data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx, 'trans': trans, 'rot': rot}
                data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
                dataset.append(data)
        # dataset.pop(0)
        return dataset

    def make_dataset_kitti_distance(self, txt_path):

            seq_list = list(range(11))

            dataset = []
            for seq in seq_list:
                dataset += (self.load_kitti_gt_txt(txt_path, seq))
            
            return dataset, seq_list

    def __getitem__(self, idx):
       
        begin = time.time()

        # idx = 216
        index_in_seq = self.dataset[idx]['anc_idx']
        index_in_seq2 = self.dataset[idx]['pos_idx']
        seq = self.dataset[idx]['seq']

        sequence = '%02d'%seq

        pc_np_file1 = os.path.join(self.train_path,'remove_outlier', sequence, '%06d.bin' % (index_in_seq))
        # dtype=np.float32应与特征点保存的格式相同，否则会出现（如double）256个特征点变成128个乱码特征点的情况
        pc_np1 = np.fromfile(pc_np_file1, dtype=np.float32)
        pc_np1 = pc_np1.reshape((-1, 4))
        pc1 = torch.tensor(pc_np1, dtype=torch.float, device=torch.device('cuda:0'))

        pc_np_file2 = os.path.join(self.train_path,'remove_outlier', sequence, '%06d.bin' % (index_in_seq2))
        pc_np2 = np.fromfile(pc_np_file2, dtype=np.float32)
        pc_np2 = pc_np2.reshape((-1, 4))
        pc2 = torch.tensor(pc_np2, dtype=torch.float, device=torch.device('cuda:0'))

        if self.remove_outlier:
            pc1 = pc1[(pc1[:,2]>-3)]
            pc2 = pc2[(pc2[:,2]>-3)]
            if self.save_remove_outlier_file:
                remove_outlier_path = '/home/chenghao/Mount/Dataset/KITTI_odometry/remove_outlier/velodyne/{}'.format(sequence)
                if not os.path.exists(remove_outlier_path):
                    os.makedirs(remove_outlier_path)
                remove_outlier = '{}/{:0>6d}.bin'.format(remove_outlier_path,index_in_seq)
                remove_outlier2 = '{}/{:0>6d}.bin'.format(remove_outlier_path,index_in_seq2)
                if not os.path.exists(remove_outlier):
                    pc1.cpu().numpy().astype(np.float32).tofile(remove_outlier)
                if not os.path.exists(remove_outlier2):
                    pc2.cpu().numpy().astype(np.float32).tofile(remove_outlier2)

        
        '''pre-extract the key points and save to files'''
        fps_path = '{}/fps/{}'.format(self.train_path, sequence)
        if not os.path.exists(fps_path):
            os.makedirs(fps_path) 
        fps_txt = '{}/{:0>6d}.bin'.format(fps_path,index_in_seq)
        fps2_txt = '{}/{:0>6d}.bin'.format(fps_path,index_in_seq2)

        if not os.path.exists(fps_txt):
            cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                pc1[None,:,:3].contiguous(), 2048
            ).long()[0]
            kp1 = pc1[cur_pt_idxs,:]
            kp1.cpu().numpy().astype(np.float32).tofile(fps_txt)
            
            # pc1.cpu().numpy().astype(np.float32).tofile(remove_outlier)

            # kp_np_file1 = os.path.join(self.train_path,'fps', sequence, '%06d.bin' % (index_in_seq))
            # kp_np1 = np.fromfile(kp_np_file1, dtype=np.float32)
            # kp1s = kp_np1.reshape((-1, 4))
            # point_cloud_o3d3 = o3d.geometry.PointCloud()
            # point_cloud_o3d3.points = o3d.utility.Vector3dVector(kp1s[:, :3])
            # point_cloud_o3d3.translate(np.asarray([0, -120, 0]))

            # point_cloud_o3d = o3d.geometry.PointCloud()
            # point_cloud_o3d.points = o3d.utility.Vector3dVector(kp1.cpu().numpy()[:, :3])

            # point_cloud_o3d2 = o3d.geometry.PointCloud()
            # point_cloud_o3d2.points = o3d.utility.Vector3dVector(pc1.cpu().numpy()[:, :3])
            # point_cloud_o3d2.translate(np.asarray([0, 120, 0]))
            # o3d.visualization.draw_geometries([point_cloud_o3d+point_cloud_o3d2+point_cloud_o3d3])

        if not os.path.exists(fps2_txt):
            cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                pc2[None,:,:3].contiguous(), 2048
            ).long()[0]
            kp2 = pc2[cur_pt_idxs,:]
            kp2.cpu().numpy().astype(np.float32).tofile(fps2_txt)
            
            # pc2.cpu().numpy().astype(np.float32).tofile(remove_outlier2)

            # point_cloud_o3d2 = o3d.geometry.PointCloud()
            # point_cloud_o3d2.points = o3d.utility.Vector3dVector(fp.cpu().numpy()[0,:,:])
            # point_cloud_o3d2.translate(np.asarray([0, 120, 0]))

            # point_cloud_o3d = o3d.geometry.PointCloud()
            # point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1.cpu().numpy()[0, :, :3])
            # o3d.visualization.draw_geometries([point_cloud_o3d+point_cloud_o3d2])
        
        return sequence

parser = argparse.ArgumentParser(
description=' ',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--train_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry', 
    help='Path to the directory of training imgs.')

parser.add_argument(
    '--txt_path', type=str, default='/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-random-full', 
    help='Path to the directory of pairs.')

parser.add_argument(
    '--remove_outlier', type=bool, default=False, 
    help='')

parser.add_argument(
    '--save_remove_outlier_file', type=bool, default=False, 
    help='')

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')

    opt = parser.parse_args()

    train_set = SparseDataset(opt)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=1, num_workers=1, drop_last=True, pin_memory = True)
    
    for i, pred in enumerate(train_loader):
        print(pred,' ',i)