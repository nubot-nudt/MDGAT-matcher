import numpy as np
import open3d as o3d
import torch
import pykitti
import os
from pathlib import Path

# import teaserpp_python


'''If your data does not include normal information, use this function'''
def compute_LRF(pc, kp, search_range, search_radius):
    # pc_all = np.concatenate((pc, kp), axis=0)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(pc)

    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)

    xs = []
    ys = []
    zs = []
    
    for i in range(kp.shape[0]):
        [k, surface_ind, _] = pcd_tree.search_hybrid_vector_3d(kp[i], search_radius, search_range)
        if k<3 :
            xs.append([1,0,0])
            ys.append([0,1,0])
            zs.append([0,0,1])
            continue
        w, v = PCA(pc[surface_ind, :], kp[i])
        v = np.array(v[:, 2])
        z = -v if np.sum(np.dot(v, (kp[i] - pc[surface_ind, :]).T)) < 0 else v
        # normals.append(normal)

        z_pro = np.multiply(np.dot(kp[i] - pc[surface_ind, :], z.T).reshape(k,1), z.reshape(1,3))
        projection = kp[i] - pc[surface_ind, :] - z_pro

        alpha = (search_radius - np.linalg.norm(kp[i] - pc[surface_ind, :],axis=1))**2
        beta = np.dot(v, (kp[i] - pc[surface_ind, :]).T)**2
        alpha = alpha.reshape(k,1)
        beta = beta.reshape(k,1)

        x = projection * alpha * beta * (1/np.linalg.norm(np.sum(projection * alpha * beta, axis=0)))
        x = np.sum(x,axis=0)

        y = np.cross(z,x)

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs, ys, zs




def PCA(data, kp, correlation=False, sort=True):
    r, c = data.shape

    data = data - kp

    cov_data = np.dot(data.T, data)/(r-1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_data)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors



if __name__ == '__main__':

    seq_list = list([0,1,2,3,4,5,6,7,8,9,10])
    for seq in seq_list:
        

        sequence = '%02d'%seq
        search_range = 60
        search_radius = 1

        pcs = []
        folder = os.path.join( '/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', sequence)
        folder = os.listdir(folder)   
        folder.sort(key=lambda x:int(x[:-4]))
        for idx in range(len(folder)):
            # folder = os.path.join(keypoints_path, 'data_odometry_velodyne', 'numpy', '%02d'%seq, np_folder)
            # folder = os.path.join(keypoints_path, 'sequences', '%02d'%seq, 'velodyne')
            kp_path='/home/chenghao/Mount/Dataset/KITTI_odometry/keypoints_USIP/tsf_256_FPFH_16384-512-k1k16-2d-nonoise'
            kp_file = os.path.join(kp_path, sequence, folder[idx])
            pc_file = os.path.join('/home/chenghao/Mount/Dataset/KITTI_odometry/preprocess-undownsample-n8', sequence, folder[idx])
            type = '%02d-%02d'%(search_range,search_radius)

            output_file_path = os.path.join(kp_path, 'LRF2', type, sequence)
            output_file = os.path.join(output_file_path, folder[idx])
            
            output_file_path = Path(output_file_path)
            output_file_path.mkdir(exist_ok=True, parents=True)
            
            if os.path.isfile(pc_file):
                pc = np.fromfile(pc_file, dtype=np.float32)
                pc = pc.reshape((-1, 8))
                pc = pc[:, :3]

                kp = np.fromfile(kp_file, dtype=np.float32)
                kp = kp.reshape((-1, 37))
                kp = kp[:, :3]

                x,y,z = compute_LRF(pc, kp, search_range, search_radius)

                output = np.hstack([np.array(x), np.array(y), np.array(z)])
                output = output.astype(np.float32)
                output.tofile(output_file)

                print(output_file + ': %d' % output.shape[0])