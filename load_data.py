import numpy as np
import torch
import os
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import open3d as o3d 
from sklearn.neighbors import KDTree
import time
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

            data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            dataset.append(data)
    # dataset.pop(0)
    return dataset

def make_dataset_kitti_distance(txt_path, mode):
        if mode == 'train':
            seq_list = list([0,2,3,4,5,6,7])
        elif mode == 'val':
            seq_list = [9]
        elif mode == 'test':
            seq_list = [10]
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

        self.train_path = opt.train_path
        self.keypoints = opt.keypoints
        self.keypoints_path = opt.keypoints_path
        self.descriptor = opt.descriptor
        self.nfeatures = opt.max_keypoints
        self.threshold = opt.threshold
        self.ensure_kpts_num = opt.ensure_kpts_num
        self.mutual_check = opt.mutual_check
        self.memory_is_enough = opt.memory_is_enough
        self.txt_path = opt.txt_path
        self.dataset, self.seq_list = make_dataset_kitti_distance(self.txt_path, mode)

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

            '''If memory is enough, load all the data'''
            if self.memory_is_enough:
                pcs = []
                folder = os.path.join(self.keypoints_path, sequence)
                folder = os.listdir(folder)   
                folder.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(folder)):
                    file = os.path.join(self.keypoints_path, sequence, folder[idx])
                    if os.path.isfile(file):
                        pc = np.fromfile(file, dtype=np.float32)
                        pcs.append(pc)
                    else:
                        pcs.append([0])
                self.pc[sequence] = pcs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # d=[9,236,390,259,1048,171,395,296]
        # d=[259,296]
        # idx=d[idx]
        index_in_seq = self.dataset[idx]['anc_idx']
        index_in_seq2 = self.dataset[idx]['pos_idx']
        seq = self.dataset[idx]['seq']
        # trans = self.dataset[idx]['trans']
        # rot = self.dataset[idx]['rot']

        # relative_pos = self.dataset[idx]['anc_idx']

        if self.memory_is_enough:
            sequence = sequence = '%02d'%seq
            pc_np1 = self.pc[sequence][index_in_seq]

            pc_np1 = pc_np1.reshape((-1, 37))
            kp1 = pc_np1[:, :3]
            score1 = pc_np1[:, 3]
            descs1 = pc_np1[:, 4:]
            pose1 = self.pose[sequence][index_in_seq] 

            pc_np2 = self.pc[sequence][index_in_seq2]
            pc_np2 = pc_np2.reshape((-1, 37))
            kp2 = pc_np2[:, :3]
            score2 = pc_np2[:, 3]
            descs2 = pc_np2[:, 4:]
            pose2 = self.pose[sequence][index_in_seq2]

            T_cam0_velo = self.calib[sequence]
            # q = np.asarray([rot[3], rot[0], rot[1], rot[2]])
            # t = np.asarray(trans)
            # relative_pose = RigidTransform(q, t)
        else:
            sequence = '%02d'%seq
            pc_np_file1 = os.path.join(self.keypoints_path, sequence, '%06d.bin' % (index_in_seq))
            pc_np1 = np.fromfile(pc_np_file1, dtype=np.float32)

            pc_np_file2 = os.path.join(self.keypoints_path, sequence, '%06d.bin' % (index_in_seq2))
            pc_np2 = np.fromfile(pc_np_file2, dtype=np.float32)
            
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

            T_cam0_velo = self.calib[sequence]

        if self.descriptor == 'pointnet' or self.descriptor == 'pointnetmsg':
            pc_file1 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq)
            pc_file2 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq2)
            pc1 = np.fromfile(pc_file1, dtype=np.float32)
            pc2 = np.fromfile(pc_file2, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            pc2 = pc2.reshape((-1, 8))
            pc1, pc2 = torch.tensor(pc1, dtype=torch.double), torch.tensor(pc2, dtype=torch.double)
   
        if self.ensure_kpts_num:
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
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1]) 
        kp2_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp2])

        vis_registered_pointcloud = False
        if vis_registered_pointcloud:
            pc_file1 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq)
            pc_file2 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq2)
            pc1 = np.fromfile(pc_file1, dtype=np.float32)
            pc2 = np.fromfile(pc_file2, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            pc2 = pc2.reshape((-1, 8))

            kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc1]) 
            kp2_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc2])

        scores1_np = np.array(score1) 
        scores2_np = np.array(score2)

        kp1_np = torch.tensor(kp1_np, dtype=torch.double)
        pose1 = torch.tensor(pose1, dtype=torch.double)
        kp2_np = torch.tensor(kp2_np, dtype=torch.double)
        pose2 = torch.tensor(pose2, dtype=torch.double)
        T_cam0_velo = torch.tensor(T_cam0_velo, dtype=torch.double)
        T_gt = torch.einsum('ab,bc,cd,de->ae', torch.inverse(T_cam0_velo), torch.inverse(pose1), pose2, T_cam0_velo) # T_gt: transpose kp2 to kp1

        '''transform pose from cam0 to LiDAR'''
        kp1w_np = torch.einsum('ki,ij,jm->mk', pose1, T_cam0_velo, kp1_np.T)
        kp2w_np = torch.einsum('ki,ij,jm->mk', pose2, T_cam0_velo, kp2_np.T)
        
        kp1w_np = kp1w_np[:, :3]
        kp2w_np = kp2w_np[:, :3]

        vis_registered_keypoints = False
        if vis_registered_keypoints:
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(kp1w_np.numpy())
            point_cloud_o3d.paint_uniform_color([0, 1, 0])
            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp2w_np.numpy())
            point_cloud_o3d2.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

        dists = cdist(kp1w_np, kp2w_np)

        '''Find ground true keypoint matching'''
        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)
        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < self.threshold]

        '''For calculating repeatibility'''
        rep = len(min1f)

        '''
        If you got high-quality keypoints, you can set the 
        mutual_check to True, otherwise, it is better to 
        set to False
        '''
        match1, match2 = -1 * np.ones((len(kp1)), dtype=np.int16), -1 * np.ones((len(kp2)), dtype=np.int16)
        if self.mutual_check:
            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            matches = np.intersect1d(min1f, xx)

            match1[min1[matches]] = matches
            match2[matches] = min1[matches]
        else:
            match1[min1v < self.threshold] = min1f

            min2v = np.min(dists, axis=0)
            min2f = min1[min2v < self.threshold]
            match2[min2v < self.threshold] = min2f

        kp1_np = kp1_np[:, :3]
        kp2_np = kp2_np[:, :3]

        norm1, norm2 = np.linalg.norm(descs1, axis=1), np.linalg.norm(descs2, axis=1)
        norm1, norm2 = norm1.reshape(kp1_num, 1), norm2.reshape(kp2_num, 1)
        descs1, descs2  = np.multiply(descs1, 1/norm1), np.multiply(descs2, 1/norm2)

        descs1, descs2 = torch.tensor(descs1, dtype=torch.double), torch.tensor(descs2, dtype=torch.double)
        scores1_np, scores2_np = torch.tensor(scores1_np, dtype=torch.double), torch.tensor(scores2_np, dtype=torch.double)

        

        return{
            # 'skip': False,
            'keypoints0': kp1_np,
            'keypoints1': kp2_np,
            'descriptors0': descs1,
            'descriptors1': descs2,
            'scores0': scores1_np,
            'scores1': scores2_np,
            'gt_matches0': match1,
            'gt_matches1': match2,
            'sequence': sequence,
            'idx0': index_in_seq,
            # 'idx1': index_in_seq2,
            # 'pose1': pose1,
            # 'pose2': pose2,
            # 'T_cam0_velo': T_cam0_velo,
            'T_gt': T_gt,
            # 'cloud0': pc1,
            # 'cloud1': pc2,
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            'rep': rep
        } 


