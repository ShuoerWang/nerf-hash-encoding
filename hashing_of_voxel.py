import torch

# section 3
@torch.no_grad()
def spatial_hash(x, T):
    primes = [1, 2654435761, 805459861]
    result = torch.zeros_like(x)[..., 0]
    for i in range(x.shape[-1]):
        result ^= x[..., i] * primes[i]
    return result % T
    

# step 1 in fig.3
def hashing_of_voxel(max_bound, min_bound, x, N_l, T):
    # locate 8 vertices of cube
    # x: [B, 3]
    cube_size = (max_bound - min_bound)/N_l;
    # [B, 3]
    bottom_left_index = torch.floor((x - min_bound)/cube_size).int()
    bottom_left_vertex = bottom_left_index * cube_size + min_bound;
    voxel_hash_indices = [] # [B, 8] 

    vertex_0 = bottom_left_index + torch.tensor([0, 0, 0])
    vertex_1 = bottom_left_index + torch.tensor([0, 0, 1])
    vertex_2 = bottom_left_index + torch.tensor([0, 1, 0])
    vertex_3 = bottom_left_index + torch.tensor([0, 1, 1])
    vertex_4 = bottom_left_index + torch.tensor([1, 0, 0])
    vertex_5 = bottom_left_index + torch.tensor([1, 0, 1])
    vertex_6 = bottom_left_index + torch.tensor([1, 1, 0])
    vertex_7 = bottom_left_index + torch.tensor([1, 1, 1])

    voxel_hash_indices.append(spatial_hash(vertex_0, T))
    voxel_hash_indices.append(spatial_hash(vertex_1, T))
    voxel_hash_indices.append(spatial_hash(vertex_2, T))
    voxel_hash_indices.append(spatial_hash(vertex_3, T))
    voxel_hash_indices.append(spatial_hash(vertex_4, T))
    voxel_hash_indices.append(spatial_hash(vertex_5, T))
    voxel_hash_indices.append(spatial_hash(vertex_6, T))
    voxel_hash_indices.append(spatial_hash(vertex_7, T))

    return cube_size, bottom_left_vertex, torch.stack(voxel_hash_indices).T

