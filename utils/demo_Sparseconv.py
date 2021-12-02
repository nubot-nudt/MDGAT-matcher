import torch
import sparseconvnet as scn
# Use the GPU if there is one, otherwise CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def transform_points_to_voxels(points):
    import spconv
    voxel_generator = spconv.utils.VoxelGenerator(
        voxel_size=[0.05, 0.05, 0.1],
        point_cloud_range=[-70.4, -40, -3, 70.4, 40, 1],
        max_num_points=5,
        max_voxels=32000,
        full_mean=False
    )        
    # pc = np.random.uniform(-10, 10, size=[1000, 3])
    voxel_features, voxel_coords, num_points = voxel_generator.generate(points.astype(np.float32))
    voxel_features = voxel_features[:,0,:]
    voxel_coords = np.pad(voxel_coords, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    voxel_coords = torch.tensor(voxel_coords, dtype=torch.int32, device=device)
    voxel_features = torch.tensor(voxel_features, dtype=torch.float32, device = device)
    # self.shape = [80, 200, 200]
    return voxel_features, voxel_coords
        


model = scn.Sequential().add(
    scn.SparseVggNet(2, 1,
                     [['C', 8], ['C', 8], ['MP', 3, 2],
                      ['C', 16], ['C', 16], ['MP', 3, 2],
                      ['C', 24], ['C', 24], ['MP', 3, 2]])
).add(
    scn.SubmanifoldConvolution(2, 24, 32, 3, False)
).add(
    scn.BatchNormReLU(32)
).add(
    scn.SparseToDense(2, 32)
).to(device)

# output will be 10x10
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
input_layer = scn.InputLayer(2, inputSpatialSize)

msgs = [[" X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
         " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
         " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
         " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
         " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "],

        [" XXX              XXXXX      x   x     x  xxxxx  xxx ",
         " X  X  X   XXX       X       x   x x   x  x     x  x ",
         " XXX                X        x   xxxx  x  xxxx   xxx ",
         " X     X   XXX       X       x     x   x      x    x ",
         " X     X          XXXX   x   x     x   x  xxxx     x ",]]


# Create Nx3 and Nx1 vectors to encode the messages above:
locations = []
features = []
for batchIdx, msg in enumerate(msgs):
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                locations.append([y, x, batchIdx])
                features.append([1])
locations = torch.LongTensor(locations)
features = torch.FloatTensor(features).to(device)

input = input_layer([locations,features])
print('Input SparseConvNetTensor:', input)
output = model(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print('Output SparseConvNetTensor:', output)


import os
import numpy as np

pc_np_file1 = os.path.join('/home/chenghao/Mount/Dataset/KITTI_odometry/remove_outlier/01', '%06d.bin' % (1))
# dtype=np.float32应与特征点保存的格式相同，否则会出现（如double）256个特征点变成128个乱码特征点的情况
pc_np1 = np.fromfile(pc_np_file1, dtype=np.float32)
pc_np1 = pc_np1.reshape((-1, 4))
# pc1 = torch.tensor(pc_np1, dtype=torch.float, device=torch.device('cuda:0'))
voxel_features, voxel_coords = transform_points_to_voxels(pc_np1)

model = scn.Sequential(
    scn.SubmanifoldConvolution(3, 4, 16, 3, False),
    scn.BatchNormReLU(16),
    scn.SubmanifoldConvolution(3, 16, 16, 3, False),
    scn.BatchNormReLU(16)
).to(device)

model2 = scn.Sequential(
    scn.Convolution(3, 16, 32, 3, 3, False),
    scn.BatchNormReLU(32),
    scn.SubmanifoldConvolution(3, 32, 32, 3, False),
    scn.BatchNormReLU(32),
    scn.SubmanifoldConvolution(3, 32, 32, 3, False),
    scn.BatchNormReLU(32)
).to(device)

model3 = scn.Sequential(
    scn.Convolution(3, 32, 64, 3, 2, False),
    scn.BatchNormReLU(64),
    scn.SubmanifoldConvolution(3, 64, 64, 3, False),
    scn.BatchNormReLU(64),
    scn.SubmanifoldConvolution(3, 64, 64, 3, False),
    scn.BatchNormReLU(64)
).to(device)

model4 = scn.Sequential(
    scn.Convolution(3, 64, 64, 3, 2, False),
    scn.BatchNormReLU(64),
    scn.SubmanifoldConvolution(3, 64, 64, 3, False),
    scn.BatchNormReLU(64),
    scn.SubmanifoldConvolution(3, 64, 64, 3, False),
    scn.BatchNormReLU(64)
).to(device)

model_out = scn.Sequential(
    scn.Convolution(3, 64, 128, [3, 1, 1], [2,1,1], False),
    scn.BatchNormReLU(128)
).to(device)

inputSpatialSize = model.input_spatial_size(torch.LongTensor([41, 1600, 2816]))
input_layer = scn.InputLayer(3, inputSpatialSize)
input = input_layer([voxel_coords, voxel_features])
output = model(input)
output2 = model2(output)
output3 = model3(output2)
output4 = model4(output3)
output5 = model_out(output4)