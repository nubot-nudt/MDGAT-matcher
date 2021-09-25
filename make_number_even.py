import os
import numpy as np

pc_folder = "/home/chenghao/Mount/Dataset/keypoints/curvature_128_FPFH_16384-512-k1k16-2d-nonoise-nonmsTG0O61HV/02"
pc_folder_list = os.listdir(pc_folder)
for idx in range(len(pc_folder_list)):
    file = os.path.join(pc_folder,'%06d.bin'%(idx))
    pc_np = np.fromfile(file, dtype=np.float32)
    pc_np=  pc_np.reshape(-1,37)
    pc_np = pc_np[:128,:]
    pc_np.tofile(os.path.join('/home/chenghao/Mount/Dataset/keypoints/curvature_128_FPFH_16384-512-k1k16-2d-nonoise-nonms/02','%06d.bin'%(idx)))

    print(idx)