import torch

# step 3 in fig.3
def trilinear_Interpolate(x, cube_size, bottle_left_vertex, voxel_hash_value):
     # voxel_hash_value : [B, 8, 2]
    top_rigtht_vertex = bottle_left_vertex + torch.tensor([1.0, 1.0, 1.0]) * cube_size
    # [B, 3]
    weights = (x - bottle_left_vertex) / (top_rigtht_vertex - bottle_left_vertex);

    # voxel_hash_value[:,0] : [B, 2]
    # weights[:, 0][:, None] : [B, 2]
    # x y z [B]
    x0 = voxel_hash_value[:, 0] * (1 - weights[:, 0])[:, None] + voxel_hash_value[:, 4] * weights[:, 0][:, None];
    x1 = voxel_hash_value[:, 1] * (1 - weights[:, 0])[:, None] + voxel_hash_value[:, 5] * weights[:, 0][:, None];
    x2 = voxel_hash_value[:, 2] * (1 - weights[:, 0])[:, None] + voxel_hash_value[:, 6] * weights[:, 0][:, None];
    x3 = voxel_hash_value[:, 3] * (1 - weights[:, 0])[:, None] + voxel_hash_value[:, 7] * weights[:, 0][:, None];

    y0 = x0 * (1 - weights[:, 1])[:, None] + x2 * weights[:, 1][:, None]
    y1 = x1 * (1 - weights[:, 1])[:, None] + x3 * weights[:, 1][:, None]

    z = y0 * (1 - weights[:, 2])[:, None] + y1 * weights[:, 2][:, None]

    return z


