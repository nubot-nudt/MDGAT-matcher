import spconv
from torch import nn
import numpy as np
import torch
class ExampleNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(3, 64, 3), # just like nn.Conv3d but don't support group and all([d > 1, s > 1])
            nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # when use submanifold convolutions, their indices can be shared to save indices generation time.
            spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(64, 64, 3, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int() # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        self.net.to(features.device)
        return self.net(x)# .dense()

voxel_generator = spconv.utils.VoxelGenerator(
    voxel_size=[0.1, 0.1, 0.1], 
    point_cloud_range=[-50, -50, -3, 50, 50, 1],
    max_num_points=30,
    max_voxels=40000,
    full_mean=False
)
points = np.random.uniform(-10, 10, size=[10000, 3])
voxel_features, voxel_coords, num_points = voxel_generator.generate(points.astype(np.float32))
voxel_features = voxel_features[:,0,:]
voxel_coords = np.pad(voxel_coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
voxel_coords = torch.tensor(voxel_coords, dtype=torch.int32)
voxel_features = torch.tensor(voxel_features, dtype=torch.float32)
# self.shape = [80, 200, 200]
net = ExampleNet([40,1000,1000])
net(voxel_features.cuda(), voxel_coords.cuda().int(), 1)
        
