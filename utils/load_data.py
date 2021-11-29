import numpy as np
import torch
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import datetime
# import PCLKeypoint

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

# sch
import pykitti
import open3d as o3d 
# from python_LOAM.src.odometry_estimator import OdometryEstimator
# from python_LOAM.src.feature_extractor import FeatureExtractor
# from python_LOAM.src.loader import Loader
from sklearn.neighbors import KDTree
import time

from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

# from autolab_core import RigidTransform
def load_kitti_gt_txt(txt_root, seq):
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

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    batchsize, ndataset, dimension = xyz.shape
    #to方法Tensors和Modules可用于容易地将对象移动到不同的设备（代替以前的cpu()或cuda()方法）
    # 如果他们已经在目标设备上则不会执行复制操作
    centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batchsize, ndataset).to(device) * 1e10
    #randint(low, high, size, dtype)
    # torch.randint(3, 5, (3,))->tensor([4, 3, 4])
    farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
    #batch_indices=[0,1,...,batchsize-1]
    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
    for i in range(npoint):
        # 更新第i个最远点
        centroids[:,i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        # 计算点集中的所有点到这个最远点的欧式距离
        #等价于torch.sum((xyz - centroid) ** 2, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        #取出每一行的最大值构成列向量，等价于torch.max(x,2)
        farthest = torch.max(distance, -1)[1]
    return centroids

def make_dataset_kitti_kframe(keypoints_path, mode):
        if mode == 'train':
            # seq_list = list(range(9))
            # seq_list = list([1, 3])
            seq_list = list([0,2,3,4,5,6,7,9])
        elif mode == 'test':
            seq_list = [10]
        else:
            raise Exception('Invalid mode.')
            
        accumulated_sample_num = 0
        sample_num_list = []
        accumulated_sample_num_list = []
        folder_list = []
        for seq in seq_list:
            # folder = os.path.join(keypoints_path, 'data_odometry_velodyne', 'numpy', '%02d'%seq, np_folder)
            # folder = os.path.join(keypoints_path, 'sequences', '%02d'%seq, 'velodyne')
            folder = os.path.join(keypoints_path, '%02d'%seq)
            folder_list.append(folder)
            
            sample_num = round(len(os.listdir(folder)))
            accumulated_sample_num += sample_num
            sample_num_list.append(sample_num)
            accumulated_sample_num_list.append(round(accumulated_sample_num))
            
        return seq_list, folder_list, sample_num_list, accumulated_sample_num_list

def make_dataset_kitti_distance(txt_path, mode):
        if mode == 'train':
            seq_list = list(range(11))
            # seq_list = list([0,2,3,4,5,6,7])
            # seq_list = list([1, 3])
            # seq_list = list([1])
        elif mode == 'val':
            seq_list = [9]
            # seq_list = range(10)
        elif mode == 'test':
            seq_list = [10]
            # seq_list = range(10)

            ## test for repeatablity
            # seq_list = [8]
        else:
            raise Exception('Invalid mode.')

        dataset = []
        for seq in seq_list:
            dataset += (load_kitti_gt_txt(txt_path, seq))
           
        return dataset, seq_list

class SparseDataset(Dataset):
    """Sparse correspondences dataset.  
    Reads images from files and creates pairs. It generates keypoints, 
    descriptors and ground truth matches which will be used in training."""

    def __init__(self, opt, mode, cfg):

        self.point_cloud_range = np.array(cfg.POINT_CLOUD_RANGE)
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(cfg.DATA_PROCESSOR[1].VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.num_point_features = cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES
        self.voxel_size = cfg.DATA_PROCESSOR[1].VOXEL_SIZE 
        self.max_num_points = cfg.DATA_PROCESSOR[1].MAX_POINTS_PER_VOXEL
        self.max_voxels = cfg.DATA_PROCESSOR[1].MAX_NUMBER_OF_VOXELS[mode]



        # self.files = []
        # self.files += [train_path + f for f in os.listdir(train_path)]
        # # os.listdir出来的文件顺序是乱的，需要重新排序
        # self.files.sort()
        self.train_path = opt.train_path
        self.keypoints = opt.keypoints

        self.keypoints_path = opt.keypoints_path
        
        self.preprocessed_path = opt.preprocessed_path
        self.descriptor = opt.descriptor

        self.nfeatures = opt.max_keypoints
        # self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)

        self.train_mode = opt.train_mode

        self.threshold = opt.threshold

        self.ensure_kpts_num = opt.ensure_kpts_num

        self.mutual_check = opt.mutual_check

        self.mode = mode

        self.memory_is_enough = opt.memory_is_enough
        self.RotAug = opt.rotation_augment
 
        if self.train_mode == 'kframe':
            self.seq_list, self.folder_list, self.sample_num_list, self.accumulated_sample_num_list = make_dataset_kitti_kframe(self.keypoints_path, mode)
        elif self.train_mode == 'distance':
            self.txt_path = opt.txt_path
            self.dataset, self.seq_list = make_dataset_kitti_distance(self.txt_path, mode)
        else:
            raise Exception('Invalid train_mode.')
        
        self.calib={}
        self.pose={}
        self.pc = {}
        self.lrf = {}

        for seq in self.seq_list:
            sequence = '%02d'%seq
            calibpath = os.path.join(self.train_path, 'calib/sequences', sequence, 'calib.txt')
            posepath = os.path.join(self.train_path, 'poses', '%02d.txt'%seq)
            with open(calibpath, 'r') as f:
                for line in f.readlines():
                    _, value = line.split(':', 1)
                    # The only non-float values in these files are dates, which
                    # we don't care about anyway
                    try:
                        calib = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
                    calib = np.reshape(calib, (3, 4))    
                    self.calib[sequence] = np.vstack([calib, [0, 0, 0, 1]])
            
            poses = []
            with open(posepath, 'r') as f:
                for line in f.readlines():
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
                self.pose[sequence] = poses
        
            if self.memory_is_enough:
                pcs = []
                lrfs= []
                folder = os.path.join(self.train_path,'preprocess-undownsample-n8', sequence)
                folder = os.listdir(folder)   
                folder.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(folder)):
                    # folder = os.path.join(keypoints_path, 'data_odometry_velodyne', 'numpy', '%02d'%seq, np_folder)
                    # folder = os.path.join(keypoints_path, 'sequences', '%02d'%seq, 'velodyne')
                    file = os.path.join(self.keypoints_path, sequence, folder[idx])
                    lrf_file = os.path.join(self.keypoints_path,'LRF/60-01', sequence, folder[idx])
                    if os.path.isfile(file):
                        pc = np.fromfile(file, dtype=np.float32)
                        lrf = np.fromfile(lrf_file, dtype=np.float32)
                        pcs.append(pc)
                        lrfs.append(lrf)
                        # x = pc.reshape((-1, 37))
                        # if x.shape[0] == 256:
                        #     print(x)
                    else:
                        pcs.append([0])
                self.pc[sequence] = pcs
                self.lrf[sequence] = lrfs
    
    def transform_points_to_voxels(self, points):
        from spconv.pytorch.utils import PointToVoxel

        voxel_generator = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel=self.max_num_points,
            max_num_voxels=self.max_voxels,
            device=points.device
        )        
        voxel_output = voxel_generator(points)

        voxels, coordinates, num_points = voxel_output

        # data_dict={}
        
        # if not data_dict['use_lead_xyz']:
        #     voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        # data_dict['points'] = points
        return voxels.cpu(), coordinates.cpu(), num_points.cpu()

    def __len__(self):
        if self.train_mode == 'kframe':
            # -1是为了不使用最后一帧数据
            # 任何超出+skipframe后超出范围的都会，同最后一帧比较
            return self.accumulated_sample_num_list[-1] - 1
        elif self.train_mode == 'distance':
            return len(self.dataset)
        else:
            raise Exception('Invalid train_mode.')

    def __getitem__(self, idx):
       
        begin = time.time()

        # idx = 216
        index_in_seq = self.dataset[idx]['anc_idx']
        index_in_seq2 = self.dataset[idx]['pos_idx']
        seq = self.dataset[idx]['seq']

        ## test for repeatablity
        # index_in_seq = 2517
        # index_in_seq2 = 3862
        # seq = self.dataset[idx]['seq']

        # trans = self.dataset[idx]['trans']
        # rot = self.dataset[idx]['rot']

        # relative_pos = self.dataset[idx]['anc_idx']

        preparetime = time.time()

        if self.memory_is_enough:
            sequence = sequence = '%02d'%seq
            pc_np1 = self.pc[sequence][index_in_seq]
            lrf1 = self.lrf[sequence][index_in_seq]
            if self.keypoints == 'sharp' or self.keypoints == 'lessharp':
                pc_np1 = pc_np1.reshape((-1, 4))
                pc_np1 = pc_np1[np.argsort(pc_np1[:, 3])[::-1]]  
                kp1 = pc_np1[:, :3]
                kp1 = kp1[:, [2,0,1]] # curvature
            else:
                pc_np1 = pc_np1.reshape((-1, 37))
                kp1 = pc_np1[:, :3]
                lrf1 = lrf1.reshape((-1,9))
                x1 = lrf1[:,:3]
                y1 = lrf1[:,3:6]
                z1 = lrf1[:,6:9]
                lrf1 = np.stack((x1,y1,z1),axis=2)
            score1 = pc_np1[:, 3]
            descs1 = pc_np1[:, 4:]
            # pose1 = dataset.poses[index_in_seq]
            pose1 = self.pose[sequence][index_in_seq] 

            pc_np2 = self.pc[sequence][index_in_seq2]
            lrf2 = self.lrf[sequence][index_in_seq2]
            if self.keypoints == 'sharp' or self.keypoints == 'lessharp':
                pc_np2 = pc_np2.reshape((-1, 4))
                pc_np2 = pc_np2[np.argsort(pc_np2[:, 3])[::-1]]
                kp2 = pc_np2[:, :3]
                kp2 = kp2[:, [2,0,1]] # curvature
                
            else:
                pc_np2 = pc_np2.reshape((-1, 37))
                kp2 = pc_np2[:, :3]
                lrf2 = lrf2.reshape((-1,9))
                x2 = lrf2[:,:3]
                y2 = lrf2[:,3:6]
                z2 = lrf2[:,6:9]
                lrf2 = np.stack((x2,y2,z2),axis=2)
            score2 = pc_np2[:, 3]
            descs2 = pc_np2[:, 4:]
            # pose2 = dataset.poses[index_in_seq2]
            pose2 = self.pose[sequence][index_in_seq2]

            T_cam0_velo = self.calib[sequence]

            # if pc_np2.shape[0]==256 or pc_np1.shape[0]==256:
            #     print('1')

            # q = np.asarray([rot[3], rot[0], rot[1], rot[2]])
            # t = np.asarray(trans)
            # relative_pose = RigidTransform(q, t)
        else:
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

            pose1 = self.pose[sequence][index_in_seq]
            pose2 = self.pose[sequence][index_in_seq2]
            T_cam0_velo = self.calib[sequence]

            '''FPS'''
            if self.mode == 'train' or self.mode == 'val':
                '''Use pre-extracted key point'''
                kp_np_file1 = os.path.join(self.train_path,'fps', sequence, '%06d.bin' % (index_in_seq))
                kp_np1 = np.fromfile(kp_np_file1, dtype=np.float32)
                kp1 = kp_np1.reshape((-1, 4))

                kp_np_file2 = os.path.join(self.train_path,'fps', sequence, '%06d.bin' % (index_in_seq2))
                kp_np2 = np.fromfile(kp_np_file2, dtype=np.float32)
                kp2 = kp_np2.reshape((-1, 4))

                kp1 = torch.tensor(kp1, dtype=torch.float, device=torch.device('cuda:0'))
                kp2 = torch.tensor(kp2, dtype=torch.float, device=torch.device('cuda:0'))
            elif self.mode == 'test':
                '''FPS the key point in real-time'''
                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                    pc1[None,:,:3].contiguous(), 2048
                ).long()[0]
                kp1 = pc1[cur_pt_idxs,:]

                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                    pc2[None,:,:3].contiguous(), 2048
                ).long()[0]
                kp2 = pc2[cur_pt_idxs,:]

        '''PointToVoxel'''
        voxels1, v_coordinates1, v_num_points1 = self.transform_points_to_voxels(pc1)
        voxels2, v_coordinates2, v_num_points2 = self.transform_points_to_voxels(pc2)

        vis_pointcloud = False
        if vis_pointcloud:
            # 显示预处理后的原始点云

            voxels = voxels1
            coordinates = v_coordinates1
            num_points = v_num_points1

            apoints = voxels.sum(1)/num_points[:,None]

            point_cloud_o3d3 = o3d.geometry.PointCloud()
            point_cloud_o3d3.points = o3d.utility.Vector3dVector(kp1[:, :3].cpu().numpy())
            point_cloud_o3d3.translate(np.asarray([0, 240, 0]))

            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(apoints[:,:3].cpu().numpy())
            point_cloud_o3d2.translate(np.asarray([0, 120, 0]))

            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1[:, :3].cpu().numpy())
            # point_cloud_o3d.normals = o3d.utility.Vector3dVector(pc1[:, 3:6])
            o3d.visualization.draw_geometries([point_cloud_o3d+point_cloud_o3d2+point_cloud_o3d3])

        vis_keypoints = False
        if vis_keypoints:
            # 显示特征点
            # pc1 = pc1.reshape((-1, 8))
            pc3_path = os.path.join(self.preprocessed_path, sequence, '%06d.bin'%(index_in_seq2))
            pc2_path = os.path.join(self.train_path, 'kitti_randomsample_16384_n8',sequence, '%06d.bin'%(index_in_seq2))
            pc2 = np.fromfile(pc2_path, dtype=np.float32)
            pc2 = pc2.reshape((-1, 8))
            pc3 = np.fromfile(pc3_path, dtype=np.float32)
            pc3 = pc3.reshape((-1, 8))
            # pc2 = pc2.reshape((-1, 4))
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(pc2[:, :3])
            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp2[:, :3])
            point_cloud_o3d3 = o3d.geometry.PointCloud()
            point_cloud_o3d3.points = o3d.utility.Vector3dVector(pc3[:, :3]+[150,0,0])
            point_cloud_o3d2.paint_uniform_color([1, 0, 0])
            point_cloud_o3d.paint_uniform_color([0, 0, 0])
            point_cloud_o3d3.paint_uniform_color([0, 0, 0])

            o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2, point_cloud_o3d3])
        
        readtime = time.time()

        '''Calculate Ground True Matches'''
        ones = torch.ones(2048, device=torch.device('cuda:0'))
        kp1h = torch.cat((kp1[:,:3], ones[:,None]), dim=1)
        kp2h = torch.cat((kp2[:,:3], ones[:,None]), dim=1)

        vis_registered_pointcloud = False
        vis_registered_keypoints = False
        if vis_registered_pointcloud:
            # pc1_path = os.path.join(self.preprocessed_path, sequence, '%06d.bin'%index_in_seq)
            # pc1 = np.fromfile(pc1_path, dtype=np.float32)
            # pc1 = pc1.reshape((-1, 8))
            # pc2_path = os.path.join(self.preprocessed_path, sequence, '%06d.bin'%index_in_seq2)
            # pc2 = np.fromfile(pc2_path, dtype=np.float32)
            # pc2 = pc2.reshape((-1, 8))

            pc_file1 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq)
            pc_file2 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq2)
            pc1 = np.fromfile(pc_file1, dtype=np.float32)
            pc2 = np.fromfile(pc_file2, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            pc2 = pc2.reshape((-1, 8))

            kp1 = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc1]) # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
            kp2 = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc2])

        pose1 = torch.tensor(pose1, dtype=torch.float, device=torch.device('cuda:0'))
        pose2 = torch.tensor(pose2, dtype=torch.float, device=torch.device('cuda:0'))
        T_cam0_velo = torch.tensor(T_cam0_velo, dtype=torch.float, device=torch.device('cuda:0'))

        # pose是cam0的轨迹真值，需将其转换到velodyne坐标系
        kp1w = torch.einsum('ki,ij,jm->mk', pose1, T_cam0_velo, kp1h.T)
        kp2w = torch.einsum('ki,ij,jm->mk', pose2, T_cam0_velo, kp2h.T)

        kp1w = kp1w[:, :3]
        kp2w = kp2w[:, :3]
        transtime = time.time()

        if vis_registered_keypoints or vis_registered_pointcloud:
            # 可视化，校准后的特征点
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(kp1w_np.numpy())
            point_cloud_o3d.paint_uniform_color([0, 1, 0])
            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp2w_np.numpy())
            point_cloud_o3d2.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

        # 计算距离
        dists = cdist(kp1w.cpu(), kp2w.cpu())

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < self.threshold]

        # 用于计算repeatibility
        rep = len(min1f)

        match1, match2 = -1 * np.ones((len(kp1)), dtype=np.int16), -1 * np.ones((len(kp2)), dtype=np.int16)
        if self.mutual_check:
            # 距离kp1最近的点是kp2，距离kp2最近的点也是kp1
            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            # 返回两个数组中共同的元素
            matches = np.intersect1d(min1f, xx)

            # setdiff1d返回数组1中2没有的元素
            # missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
            # missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

            # match1, match2 = -1 * np.ones((len(kp1)), dtype=np.int16), -1 * np.ones((len(kp2)), dtype=np.int16)
            match1[min1[matches]] = matches
            match2[matches] = min1[matches]
        else:
            match1[min1v < self.threshold] = min1f

            min2v = np.min(dists, axis=0)
            min2f = min1[min2v < self.threshold]
            match2[min2v < self.threshold] = min2f

        gttime = time.time()

        

        ''' augment training data with random rotation'''
        if self.RotAug == True:
            theta=np.random.rand(1)*2*np.pi#0到2*pi的均匀分布
            R_z = np.array([[math.cos(theta),    -math.sin(theta),    0],
                    [math.sin(theta),    math.cos(theta),     0],
                    [0,                     0,                      1]
                    ])
            Rt_z = np.array([[math.cos(theta),    -math.sin(theta),    0, 0],
                    [math.sin(theta),    math.cos(theta),     0, 0],
                    [0,                     0,                      1, 0],
                    [0,0,0,1]
                    ])
            R_z = torch.tensor(R_z, dtype=torch.float, device=torch.device('cuda:0'))
            Rt_z = torch.tensor(Rt_z, dtype=torch.float, device=torch.device('cuda:0'))
            r_kp1 = torch.einsum('ki,ji->jk', R_z, kp1[:, :3])
            kp1 = torch.cat((r_kp1,kp1[:,3][:,None]), dim=1)
        else:
            Rt_z = np.array([[1, 0, 0, 0],
                            [0, 1,  0, 0],
                            [0, 0,  1, 0],
                            [0, 0,  0,  1]
                            ])
            Rt_z = torch.tensor(Rt_z, dtype=torch.float, device=torch.device('cuda:0'))


        backendtime = time.time()
        
        # print('preparetime {}, readtime {}, transtime: {}, gttime: {}, backendtime: {}' 
                        # .format(preparetime-begin, readtime-preparetime, transtime-readtime, gttime-transtime, backendtime-gttime)) 

        return{
            # 'skip': False,
            'keypoints0': kp1.cpu(),
            'keypoints1': kp2.cpu(),
            'voxels0': voxels1,
            'voxels1': voxels2,
            'voxel_coords0': v_coordinates1,
            'voxel_coords1': v_coordinates2,
            'voxel_num_points0': v_num_points1,
            'voxel_num_points1': v_num_points2,
            'match0': match1,
            'match1': match2,
            'sequence': sequence,
            'idx0': index_in_seq,
            'idx1': index_in_seq2,
            # 'repeat': min1f
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            'rep': rep,
            'Rt_z': Rt_z.cpu()
        } 
            

