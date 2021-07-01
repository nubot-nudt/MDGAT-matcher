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
            # seq_list = list(range(9))
            seq_list = list([0,2,3,4,5,6,7])
            # seq_list = list([1, 3])
            # seq_list = list([1])
        elif mode == 'val':
            seq_list = [9]
            # seq_list = range(10)
        elif mode == 'test':
            seq_list = [10]
            # seq_list = range(10)
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

    def __init__(self, opt, mode):

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

        # 训练点云间隔帧
        self.kframe = opt.kframe
        self.train_mode = opt.train_mode

        self.threshold = opt.threshold

        self.ensure_kpts_num = opt.ensure_kpts_num

        self.mutual_check = opt.mutual_check

        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        # 暴力匹配
        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

        # 角点提取
        # self.extractor = FeatureExtractor()
        # self.N_SCANS = 64
        self.memory_is_enough = opt.memory_is_enough
 
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
                folder = os.path.join(self.train_path,'preprocess-undownsample-n8', sequence)
                folder = os.listdir(folder)   
                folder.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(folder)):
                    # folder = os.path.join(keypoints_path, 'data_odometry_velodyne', 'numpy', '%02d'%seq, np_folder)
                    # folder = os.path.join(keypoints_path, 'sequences', '%02d'%seq, 'velodyne')
                    file = os.path.join(self.keypoints_path, sequence, folder[idx])
                    if os.path.isfile(file):
                        pc = np.fromfile(file, dtype=np.float32)
                        pcs.append(pc)
                        x = pc.reshape((-1, 37))
                        if x.shape[0] == 256:
                            print(x)
                    else:
                        pcs.append([0])
                self.pc[sequence] = pcs

    def __len__(self):
        if self.train_mode == 'kframe':
            # -1是为了不使用最后一帧数据
            # 任何超出+skipframe后超出范围的都会，同最后一帧比较
            return self.accumulated_sample_num_list[-1] - 1
        elif self.train_mode == 'distance':
            return len(self.dataset)
        else:
            raise Exception('Invalid train_mode.')

    # ISS
    def _get_scan_ids(self, pcd):
        depth = np.linalg.norm(pcd[:, :3], 2, axis=1)
        pitch = np.arcsin(pcd[:, 2] / depth)
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        scan_ids = (pitch + abs(fov_down)) / fov
        scan_ids *= self.N_SCANS
        scan_ids = np.floor(scan_ids)
        scan_ids = np.minimum(self.N_SCANS - 1, scan_ids)
        scan_ids = np.maximum(0, scan_ids).astype(np.int32)
        return scan_ids
    # ISS
    def reorder_pcd(self, pcd):
        scan_start = np.zeros(self.N_SCANS, dtype=int)
        scan_end = np.zeros(self.N_SCANS, dtype=int)

        scan_ids = self._get_scan_ids(pcd)
        sorted_ind = np.argsort(scan_ids, kind='stable')
        sorted_pcd = pcd[sorted_ind]
        sorted_scan_ids = scan_ids[sorted_ind]

        elements, elem_cnt = np.unique(sorted_scan_ids, return_counts=True)

        start = 0
        for ind, cnt in enumerate(elem_cnt):
            scan_start[ind] = start
            start += cnt
            scan_end[ind] = start

        laser_cloud = np.hstack((sorted_pcd, sorted_scan_ids.reshape((-1, 1))))
        return laser_cloud, scan_start, scan_end

    def __getitem__(self, idx):
        iss = False
        if iss :
            # 提取pc1,pose1,timestamp1,kp1
            dataset = pykitti.odometry(self.train_path, self.sequence)
            pc1 = dataset.get_velo(idx)
            pose1 = dataset.poses[idx]
            timestamp1 = dataset.timestamps[idx]

            # 当前帧没有在它kframe帧之后的帧用于匹配、训练
            if idx + self.kframe > len(self) - 1:
                return{
                    'keypoints0': torch.zeros([0, 0, 3], dtype=torch.double),
                    'keypoints1': torch.zeros([0, 0, 3], dtype=torch.double),
                    'descriptors0': torch.zeros([0, 3], dtype=torch.double),
                    'descriptors1': torch.zeros([0, 3], dtype=torch.double),
                    # 'cloud0': pc1,
                    # 'cloud1': pc1,
                    # 加上timastamp会报错
                    # 'timestamp0': timestamp1,
                    # 'timestamp1': timestamp1,
                    'idx': idx
                } 

            # 降采样
            start = time.time()
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1[:, :3])
            point_cloud_o3d = point_cloud_o3d.voxel_down_sample(voxel_size=0.01)
            point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            print('downsample and normals: ', time.time() - start)
            points = np.asarray(point_cloud_o3d.points)
            normals = np.asarray(point_cloud_o3d.normals)
        
            # points = self.reorder_pcd(points)
            # sharp_points, sharp_ind = self.extractor.extract_features(points[0], points[1], points[2])
            # kp1 = np.asarray(sharp_points)

            # ISS特征提取
            start = time.time()
            feature_idx, FPFH = iss_descriptor(points, normals)
            kp1 = np.asarray(pc1[feature_idx])
            descs1 = FPFH
            print('ISS and FPFH: ', time.time() - start)
        

            # 特征点可视化
            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp1[:256, :3])
            point_cloud_o3d2.paint_uniform_color([255, 0, 0])
            point_cloud_o3d.paint_uniform_color([0, 0, 0])
            o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])
            
            # 跳过kframe，作为目标点云
            idx += self.kframe
            # 提取pc2,pose2,timestamp2,kp2
            pc2 = dataset.get_velo(idx)
            pose2 = dataset.poses[idx]
            timestamp2 = dataset.timestamps[idx]
            
            start = time.time()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(pc2[:, :3])
            point_cloud_o3d.voxel_down_sample(voxel_size=0.005)
            point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

            points = np.asarray(point_cloud_o3d.points)
            normals = np.asarray(point_cloud_o3d.normals)

            # points = self.reorder_pcd(points[:, :3])
            # sharp_points, sharp_ind = self.extractor.extract_features(points[0], points[1], points[2])
            # kp2 = np.asarray(sharp_points)  
            
            # ISS特征提取
            feature_idx, FPFH = iss_descriptor(points, normals)
            kp2 = np.asarray(pc2[feature_idx])
            descs2 = FPFH
            print('2:downsample+normals ISS and FPFH: ', time.time() - start)

            # #特征点，描述子生成
            if len(kp1) < 1 or len(kp2) < 1:
                # print("no kp: ",file_name)
                return{
                    'keypoints0': torch.zeros([0, 0, 3], dtype=torch.double),
                    'keypoints1': torch.zeros([0, 0, 3], dtype=torch.double),
                    'descriptors0': torch.zeros([0, 3], dtype=torch.double),
                    'descriptors1': torch.zeros([0, 3], dtype=torch.double),
                    # 'cloud0': pc1,
                    # 'cloud1': pc2,
                    # 'timestamp0': timestamp1,
                    # 'timestamp1': timestamp2,
                    'idx': idx
                }
                
        begin = time.time()
        if self.train_mode == 'kframe':
            for i, accumulated_sample_num in enumerate(self.accumulated_sample_num_list):
                if idx < accumulated_sample_num:
                    break
            folder = self.folder_list[i]
            seq = self.seq_list[i]

            if i == 0:
                index_in_seq = idx
            else:
                index_in_seq = idx - self.accumulated_sample_num_list[i-1]
            
            # 如果+kframe超出索引，那么就指向最后一帧点云
            index_in_seq2 = index_in_seq + self.kframe
            if index_in_seq2 > self.sample_num_list[i] - 1:
                index_in_seq2 = self.sample_num_list[i] - 1
        elif self.train_mode == 'distance':
            # idx = 216
            index_in_seq = self.dataset[idx]['anc_idx']
            index_in_seq2 = self.dataset[idx]['pos_idx']
            seq = self.dataset[idx]['seq']
            # trans = self.dataset[idx]['trans']
            # rot = self.dataset[idx]['rot']

            # relative_pos = self.dataset[idx]['anc_idx']
        else:
            raise Exception('Invalid train_mode.')
        preparetime = time.time()

        if self.memory_is_enough:
            sequence = sequence = '%02d'%seq
            pc_np1 = self.pc[sequence][index_in_seq]
            if self.keypoints == 'sharp' or self.keypoints == 'lessharp':
                pc_np1 = pc_np1.reshape((-1, 4))
                pc_np1 = pc_np1[np.argsort(pc_np1[:, 3])[::-1]]  
                kp1 = pc_np1[:, :3]
                kp1 = kp1[:, [2,0,1]] # curvature
            else:
                pc_np1 = pc_np1.reshape((-1, 37))
                kp1 = pc_np1[:, :3]
            score1 = pc_np1[:, 3]
            descs1 = pc_np1[:, 4:]
            # pose1 = dataset.poses[index_in_seq]
            pose1 = self.pose[sequence][index_in_seq] 

            pc_np2 = self.pc[sequence][index_in_seq2]
            if self.keypoints == 'sharp' or self.keypoints == 'lessharp':
                pc_np2 = pc_np2.reshape((-1, 4))
                pc_np2 = pc_np2[np.argsort(pc_np2[:, 3])[::-1]]
                kp2 = pc_np2[:, :3]
                kp2 = kp2[:, [2,0,1]] # curvature
            else:
                pc_np2 = pc_np2.reshape((-1, 37))
                kp2 = pc_np2[:, :3]
            score2 = pc_np2[:, 3]
            descs2 = pc_np2[:, 4:]
            # pose2 = dataset.poses[index_in_seq2]
            pose2 = self.pose[sequence][index_in_seq2]

            T_cam0_velo = self.calib[sequence]

            if pc_np2.shape[0]==256 or pc_np1.shape[0]==256:
                print('1')

            # q = np.asarray([rot[3], rot[0], rot[1], rot[2]])
            # t = np.asarray(trans)
            # relative_pose = RigidTransform(q, t)
        else:
            sequence = '%02d'%seq
            # dataset = pykitti.odometry(self.train_path, sequence)
            pc_np_file1 = os.path.join(self.keypoints_path, sequence, '%06d.bin' % (index_in_seq))
            # dtype=np.float32应与特征点保存的格式相同，否则会出现（如double）256个特征点变成128个乱码特征点的情况
            pc_np1 = np.fromfile(pc_np_file1, dtype=np.float32)

            pc_np_file2 = os.path.join(self.keypoints_path, sequence, '%06d.bin' % (index_in_seq2))
            pc_np2 = np.fromfile(pc_np_file2, dtype=np.float32)
            
            
            if self.keypoints == 'sharp' or self.keypoints == 'lessharp':
                pc_np1 = pc_np1.reshape((-1, 4))
                pc_np1 = pc_np1[np.argsort(pc_np1[:, 3])[::-1]]
                kp1 = pc_np1[:, :3]
                kp1 = kp1[:, [2,0,1]] # curvature

                pc_np2 = pc_np2.reshape((-1, 4))
                pc_np2 = pc_np2[np.argsort(pc_np2[:, 3])[::-1]]
                kp2 = pc_np2[:, :3]
                kp2 = kp2[:, [2,0,1]] # curvature
            else:
                pc_np1 = pc_np1.reshape((-1, 37))
                kp1 = pc_np1[:, :3]

                pc_np2 = pc_np2.reshape((-1, 37))
                kp2 = pc_np2[:, :3]
                
            score1 = pc_np1[:, 3]
            descs1 = pc_np1[:, 4:]
            # pose1 = dataset.poses[index_in_seq]
            pose1 = self.pose[sequence][index_in_seq]
            # pc1 = dataset.get_velo(index_in_seq)

            score2 = pc_np2[:, 3]
            descs2 = pc_np2[:, 4:]
            # pose2 = dataset.poses[index_in_seq2]
            pose2 = self.pose[sequence][index_in_seq2]

            # debug显示nan点
            # indx = []
            # for i in range(512):
            #     if(math.isnan(descs1[i,0])):  
            #         indx.append(i)
            # pc1 = dataset.get_velo(idx)
            # point_cloud_o3d = o3d.geometry.PointCloud()
            # point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1[:, :3])
            # point_cloud_o3d2 = o3d.geometry.PointCloud()
            # point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp1[indx, :3])
            # point_cloud_o3d2.paint_uniform_color([255, 0, 0])
            # point_cloud_o3d.paint_uniform_color([0, 0, 0])
            # o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

            T_cam0_velo = self.calib[sequence]

        # @todo: 不使用随机采样 or 实时降采样
        if self.descriptor == 'pointnet' or self.descriptor == 'pointnetmsg':
            # pc_file1 = os.path.join(self.train_path, 'preprocess-undownsample-n8', sequence, '%06d.bin' % index_in_seq)
            # pc_file2 = os.path.join(self.train_path, 'preprocess-undownsample-n8', sequence, '%06d.bin' % index_in_seq2)
            # pc1 = np.fromfile(pc_file1, dtype=np.float32)
            # pc2 = np.fromfile(pc_file2, dtype=np.float32)
            # pc1 = pc1.reshape((-1, 8))
            # pc2 = pc2.reshape((-1, 8))
            # choice_idx1 = np.random.choice(pc1.shape[0], 16384, replace=False)
            # choice_idx2 = np.random.choice(pc2.shape[0], 16384, replace=False)
            # pc1 = pc1[choice_idx1, :]
            # pc2 = pc2[choice_idx2, :]
            # pc1, pc2 = torch.tensor(pc1, dtype=torch.double), torch.tensor(pc2, dtype=torch.double)

            pc_file1 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq)
            pc_file2 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq2)
            pc1 = np.fromfile(pc_file1, dtype=np.float32)
            pc2 = np.fromfile(pc_file2, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            pc2 = pc2.reshape((-1, 8))
            pc1, pc2 = torch.tensor(pc1, dtype=torch.double), torch.tensor(pc2, dtype=torch.double)

        
        vis_pointcloud = False
        if vis_pointcloud:
            # 显示预处理后的原始点云
            pc1_path = os.path.join(self.preprocessed_path, sequence, '%06d.bin'%index_in_seq)
            pc1 = np.fromfile(pc1_path, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(pc1[:, :3])
            # point_cloud_o3d.normals = o3d.utility.Vector3dVector(pc_np[:, 3:6])
            o3d.visualization.draw_geometries([point_cloud_o3d])

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

        
        
        if not self.keypoints == 'USIP' and self.ensure_kpts_num:
            # kp1_num = min(self.nfeatures, len(kp1))
            # kp2_num = min(self.nfeatures, len(kp2))
            valid1 = score1>10
            valid2 = score2>10
            kp1=kp1[valid1]
            kp2=kp2[valid2]
            score1=score1[valid1]
            score2=score2[valid2]
            descs1=descs1[valid1]
            descs2=descs2[valid2]
            kp1_num = self.nfeatures
            kp2_num = self.nfeatures
            if kp1_num < len(kp1):
                kp1 = kp1[:kp1_num]
                score1 = score1[:kp1_num]
                descs1 = descs1[:kp1_num]
            else:
                while kp1_num > len(kp1):
                    kp1 = np.vstack((kp1[:(kp1_num-len(kp1))], kp1))
                    score1 = np.hstack((score1[:(kp1_num-len(score1))], score1))
                    descs1 = np.vstack((descs1[:(kp1_num-len(descs1))], descs1))
            
            if kp2_num < len(kp2):
                kp2 = kp2[:kp2_num]
                score2 = score2[:kp2_num]
                descs2 = descs2[:kp2_num]
            else:
                while kp2_num > len(kp2):
                    kp2 = np.vstack((kp2[:(kp2_num-len(kp2))], kp2))
                    score2 = np.hstack((score2[:(kp2_num-len(score2))], score2))
                    descs2 = np.vstack((descs2[:(kp2_num-len(descs2))], descs2))
        else:
            kp1_num = len(kp1)
            kp2_num = len(kp2)
        # 转换为齐次
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1]) # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
        kp2_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp2])
        # 可视化显示原始点云
        vis_registered_pointcloud = False
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

            kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc1]) # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
            kp2_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc2])

        #  # 特征点置信度
        scores1_np = np.array(score1) 
        scores2_np = np.array(score2)

        # 根据M将特征点透视变化目标图片
        kp1_np = torch.tensor(kp1_np, dtype=torch.double)
        pose1 = torch.tensor(pose1, dtype=torch.double)
        kp2_np = torch.tensor(kp2_np, dtype=torch.double)
        pose2 = torch.tensor(pose2, dtype=torch.double)
        T_cam0_velo = torch.tensor(T_cam0_velo, dtype=torch.double)

        # relative_pose = torch.tensor(relative_pose.matrix, dtype=torch.double)

        # pose是cam0的轨迹真值，需将其转换到velodyne坐标系
        
        kp1w_np = torch.einsum('ki,ij,jm->mk', pose1, T_cam0_velo, kp1_np.T)
        kp2w_np = torch.einsum('ki,ij,jm->mk', pose2, T_cam0_velo, kp2_np.T)

        # kp1w_np = torch.einsum('ki,ij->jk', relative_pose, kp1_np.T)
        # kp2w_np = kp2_np
        
        kp1w_np = kp1w_np[:, :3]
        kp2w_np = kp2w_np[:, :3]
        transtime = time.time()

        vis_registered_keypoints = False
        if vis_registered_keypoints:
            # 可视化，校准后的特征点
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(kp1w_np.numpy())
            point_cloud_o3d.paint_uniform_color([0, 1, 0])
            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp2w_np.numpy())
            point_cloud_o3d2.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

        # 计算距离
        dists = cdist(kp1w_np, kp2w_np)

        # for mm in matched:
        #     dd = dists[mm.queryIdx, mm.trainIdx]
        #     print(dd)

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

        visualize = False
        if visualize:
            matches_dmatch = []
            for idx in range(matches.shape[0]):
                dmatch = cv2.DMatch(matches[idx], min2[matches[idx]], 0.0)
                print("Match {matches[idx]} {min2[matches[idx]]} dist={dists[matches[idx], min2[matches[idx]]]}")
                matches_dmatch.append(dmatch)
            out = cv2.drawMatches(image, kp1, warped, kp2, matches_dmatch, None)
            cv2.imshow('a', out)
            cv2.waitKey(0)

        # MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        # MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        # MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        # all_matches = np.concatenate([MN, MN2, MN3], axis=1)
        '''
        for idx in range(all_matches.shape[1]):
            pt1 = all_matches[0, idx]
            pt2 = all_matches[1, idx]
            if pt1 != self.nfeatures and pt2 != self.nfeatures:
                print(f"match: {dists[pt1, pt2]} | {pt2} {np.argmin(dists[pt1, :])} | {pt1} {np.argmin(dists[:, pt2])}")
            else:
                print(f"no match {pt1} {pt2}")
        '''
        # if kp1_np.shape != kp2_np.shape:
        #     print(kp1_np.shape, kp2_np.shape)
        #     print("MN", MN)
        #     print("MN2", MN2)
        #     print("MN3", MN3)
        #     print(" ")

        # return {'kp1': kp1_np / max_size, 'kp2': kp2_np / max_size, 'descs1': descs1 / 256., 'descs2': descs2 / 256., 'matches': all_matches}
        kp1_np = kp1_np[:, :3]
        kp2_np = kp2_np[:, :3]

        # 归一化
        norm1, norm2 = np.linalg.norm(descs1, axis=1), np.linalg.norm(descs2, axis=1)
        norm1, norm2 = norm1.reshape(kp1_num, 1), norm2.reshape(kp2_num, 1)
        descs1, descs2  = np.multiply(descs1, 1/norm1), np.multiply(descs2, 1/norm2)

        # descs1, descs2 = descs1.reshape(( -1, 33)), descs2.reshape(( -1, 33))
        descs1, descs2 = torch.tensor(descs1, dtype=torch.double), torch.tensor(descs2, dtype=torch.double)
        scores1_np, scores2_np = torch.tensor(scores1_np, dtype=torch.double), torch.tensor(scores2_np, dtype=torch.double)
        # descs1 = np.transpose(descs1 / 256.)
        # descs2 = np.transpose(descs2 / 256.)
        # kp1_np = kp1_np.reshape((1, -1, 3))
        # kp2_np = kp2_np.reshape((1, -1, 3))
        backendtime = time.time()
        
        # print('preparetime {}, readtime {}, transtime: {}, gttime: {}, backendtime: {}' 
                        # .format(preparetime-begin, readtime-preparetime, transtime-readtime, gttime-transtime, backendtime-gttime)) 

        # 归一化
        # image = torch.from_numpy(image/255.).double()[None].cuda()
        # warped = torch.from_numpy(warped/255.).double()[None].cuda()
        # print('calculate gt: ', time.time() - start)
        if self.descriptor == 'FPFH' or self.descriptor == 'FPFH_gloabal' or self.descriptor == 'FPFH_only':
            return{
                # 'skip': False,
                # 'keypoints0': list(kp1_np),
                # 'keypoints1': list(kp2_np),
                # 'descriptors0': list(descs1),
                # 'descriptors1': list(descs2),
                # 'scores0': list(scores1_np),
                # 'scores1': list(scores2_np),
                # 'match0': list(match1),
                # 'match1': list(match2),
                # 'sequence': sequence,
                # 'idx0': index_in_seq,
                # 'idx1': index_in_seq2,
                'keypoints0': kp1_np,
                'keypoints1': kp2_np,
                'descriptors0': descs1,
                'descriptors1': descs2,
                'scores0': scores1_np,
                'scores1': scores2_np,
                'match0': match1,
                'match1': match2,
                'sequence': sequence,
                'idx0': index_in_seq,
                'idx1': index_in_seq2,
                # 'repeat': min1f
                # 'cloud0': pc1,
                # 'cloud1': pc2,
                # 'all_matches': list(all_matches),
                # 'file_name': file_name
                'rep': rep
            } 
        else:
            return{
            # 'skip': False,
            # 'keypoints0': list(kp1_np),
            # 'keypoints1': list(kp2_np),
            # 'descriptors0': list(descs1),
            # 'descriptors1': list(descs2),
            # 'scores0': list(scores1_np),
            # 'scores1': list(scores2_np),
            # 'match0': list(match1),
            # 'match1': list(match2),
            # 'sequence': sequence,
            # 'idx0': index_in_seq,
            # 'idx1': index_in_seq2,
            'keypoints0': kp1_np,
            'keypoints1': kp2_np,
            'descriptors0': descs1,
            'descriptors1': descs2,
            'scores0': scores1_np,
            'scores1': scores2_np,
            'match0': match1,
            'match1': match2,
            'sequence': sequence,
            'idx0': index_in_seq,
            'idx1': index_in_seq2,
            # 'repeat': min1f
            'cloud0': pc1,
            'cloud1': pc2,
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            'rep': rep
            } 
            

