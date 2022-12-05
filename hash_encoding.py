import torch
import torch.nn as nn

from hashing_of_voxel import hashing_of_voxel
from trilinear_Interpolate import trilinear_Interpolate

class HashEncoder(nn.Module):
    def __init__(self, bounding_box):
        super(HashEncoder, self).__init__()
        self.min_bound, self.max_bound = bounding_box
        self.logT = 14;
        self.L = 16;
        self.T = 2 ** 14;
        self.F = 2;
        self.N_min =  torch.tensor(16);
        self.N_max =  torch.tensor(512);

        self.b = torch.exp((torch.log(self.N_max)-torch.log(self.N_min))/(self.L-1))
        hash_tables = [];
        for i in range(self.L):
            # [T, F]
            hash_table = nn.Embedding(self.T, self.F);
            nn.init.uniform_(hash_table.weight, a=-0.0001, b=0.0001)
            hash_tables.append(hash_table)

        self.hash_tables = nn.ModuleList(hash_tables);


    def forward(self, x):
        # x: [B, 3]
        feature_vector = [];
        for i in range(self.L):
            N_l = torch.floor(self.N_min * self.b**i)
            cube_size, bottle_left_vertex, voxel_hash_indices = hashing_of_voxel(self.max_bound, self.min_bound, x, N_l, self.T)
            # step 2 in fig.3
            voxel_hash_value = self.hash_tables[i](voxel_hash_indices)
            # step 4 in fig.3
            feature_vector.append(trilinear_Interpolate(x, cube_size, bottle_left_vertex, voxel_hash_value))

        return torch.cat(feature_vector, dim=-1)

