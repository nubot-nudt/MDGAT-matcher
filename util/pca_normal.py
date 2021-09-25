# -*- coding: utf-8 -*-
# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import shutil
import time

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    r, c = data.shape
    mean = np.mean(data, 0)

    data = data - mean

    cov_data = np.dot(data.T, data)/(r-1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_data)
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def txt2xyzn(txt_root_dir):

    # 新建存放xyzn数据集的根目录
    xyzn_root_dir = txt_root_dir + "_xyzn"
    if os.path.exists(xyzn_root_dir):
        ans = input("xyzn dataset already exits, do you want to overwrite the dataset? Please answer y or n: \n")
        while True:
            if ans == "y":
                shutil.rmtree(xyzn_root_dir)
                os.mkdir(xyzn_root_dir)
                break
            elif ans == "n":
                print("The existing xyzn dataset will be used!")
                return None
            else:
                ans = input("The input answer is wrong. Please answer y or n:")
    else:
        os.mkdir(xyzn_root_dir)

    print("Converting files ......")

    # 获取所有txt文件
    txt_files = list()
    for root, dirs, files in os.walk(txt_root_dir):
        # if root != txt_root_dir:
            for file in files:
                txt_files.append(os.path.join(root, file))

    # 用空格替换掉逗号，并且把处理后的数据保存为xyzn格式
    for txt_file in txt_files:
        xyzn_data = list()
        with open(txt_file) as f:
            for line in f:
                line = line.replace(",", " ")
                xyzn_data.append(line)
        _, txt_file_name = os.path.split(txt_file)  # 分割路径名和文件名
        name, extension = os.path.splitext(txt_file_name)  # 分割文件名里面的名字以及后缀
        txt_class = txt_file_name.split("_")[0]
        xyzn_class_dir = os.path.join(xyzn_root_dir, txt_class)
        if not os.path.exists(xyzn_class_dir):  # 在xyzn根目录下，以类别作为子路径名，新建子路径
            os.mkdir(xyzn_class_dir)
        xyzn_file_name = name + ".xyzn"
        xyzn_file = os.path.join(xyzn_class_dir, xyzn_file_name)
        with open(xyzn_file, "a") as f:  # 保存xyzn文件
            for each_data in xyzn_data:
                f.write(each_data)

    print("Converting process is done!")

def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    txt_root_dir = "/media/nubot/新加卷1/施成浩ws/三维点云处理网课/test"
    ob = "desk"
    xyzn_root_dir = txt_root_dir + "_xyzn"

    # 加载原始点云
    point_cloud_o3d = o3d.io.read_point_cloud(xyzn_root_dir+"/"+ob+"/"+ob+"_0001.xyzn")
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云  
    
    
    # 从点云中获取点，只对点进行处理
    points = np.asarray(point_cloud_o3d.points)
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    tic = time.time()
    w, pca = PCA(points)
    toc = time.time()
    print("PCA time: ", toc - tic)
    
    point_cloud_vector = pca[:, 0] #点云主方向对应的向量
    print('the main orientation of this pointcloud is(原来取v[:, 2]有误,主方向应是特征值最大的方向,即v[:, 0]): ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA

    # test: 投影至主方向
    # point_cloud_projected = np.dot(points, point_cloud_vector)
    # point_cloud_projected = np.dot(point_cloud_projected, point_cloud_vector.T)
    # point_cloud_projected_o3d = o3d.geometry.PointCloud() 
    # point_cloud_projected_o3d.points = o3d.utility.Vector3dVector(point_cloud_projected)
    # o3d.visualization.draw_geometries([point_cloud_projected_o3d])
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    search_range = 20
    search_radius = 0.2
    
    tic = time.time()
    for i in range(points.shape[0]):
        [k, surface_ind, _] = pcd_tree.search_hybrid_vector_3d(points[i], search_radius, search_range) 
        w, v = PCA(points[surface_ind, :])
        normals.append(v[:, 2]) 
    toc = time.time()
    print("normal time: ", toc - tic)
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    
    # test: 投影PCA
    points = [[0, 0, 0], pca[:, 0], pca[:, 1], pca[:, 2]]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([point_cloud_o3d]+[line_set])


if __name__ == '__main__':
    main()
