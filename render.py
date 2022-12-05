import torch
import imageio
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

num_samples = 64
ray_chunk = 1024*32

def inference(points, dirs, model):
    points_vector = torch.reshape(points, [-1, points.shape[-1]])
    dir_expand = dirs[:,None].expand(points.shape)
    dirs_vector = torch.reshape(dir_expand, [-1, dir_expand.shape[-1]])

    outputs_vector = model(points_vector, dirs_vector)
    outputs = torch.reshape(outputs_vector, list(points.shape[:-1]) + [outputs_vector.shape[-1]])
    return outputs

# convert from (0, 1) value to 8 bit rgb value
def convert28bit(value):
    return (255 * np.clip(value, 0, 1)).astype(np.uint8)

# convert network output to rgs values
def output2rgb(output, rays, ray_dir):
    ray_distance = rays[...,1:] - rays[...,:-1]
    ray_distance = torch.cat([ray_distance, torch.Tensor([1e10]).expand(ray_distance[...,:1].shape)], -1)
    ray_distance *= torch.norm(ray_dir[...,None,:], dim=-1)

    alpha = 1. - torch.exp(-F.relu(output[...,3])*ray_distance)
    alpha_shift = torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-15], dim = -1)

    weights = alpha * torch.cumprod(alpha_shift, dim = -1)[:, :-1]
    opcacity = torch.sum(weights, -1)

    rgb = torch.sigmoid(output[...,:3])
    rgbs = torch.sum(weights[...,None] * rgb, -2)
    rgbs = rgbs + (1. - opcacity[...,None])
    return rgbs

# generate rays by photo information
def get_rays(hwf, cam2world):
    height, width, focal = hwf
    height_vector = torch.linspace(0, height -1 , height)
    width_vector = torch.linspace(0, width - 1, width)
    i, j = torch.meshgrid(width_vector, height_vector) 
    i = i.t()
    j = j.t()
    width_angle = (i - (0.5 * width))/focal
    height_angle = - (j - 0.5 * height)/focal
    dirs = torch.stack([width_angle, height_angle, -torch.ones_like(i)], -1)

    rays_d = torch.sum(dirs[..., np.newaxis, :] * cam2world[:3,:3], -1)
    rays_o = cam2world[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# function for inference and get rgb colors for rays
def render(rays, model, random_sample):
    rgbs = []
    for i in range(0, rays.shape[0], ray_chunk):
        batch = rays[i: i + ray_chunk]

        # [B, 3]
        rays_o, rays_d, dirs = batch[:,0:3], batch[:,3:6], batch[:,6:9]
        # parameters of blender format
        near = 2. * torch.ones_like(rays_d[...,:1])
        far = 6. * torch.ones_like(rays_d[...,:1])

        t_vals = torch.linspace(0., 1., steps=num_samples)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.expand([batch.shape[0], num_samples])

        if random_sample:
            middle = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([middle, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], middle], -1)
            random = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * random

        points = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        output = inference(points, dirs, model)
        rgb = output2rgb(output, z_vals, rays_d)
        rgbs.append(rgb)

    return torch.cat(rgbs, 0)

# render given rays
def render_rays(model, random_sample, hwf, rays=None, cam2world=None):
    if random_sample:
        rays_o, rays_d = rays
    else:
        rays_o, rays_d = get_rays(hwf, cam2world)

    dirs = rays_d
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs = torch.reshape(dirs, [-1,3]).float()

    origin_shape = rays_d.shape
    # [B, 3]
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    rays = torch.cat([rays_o, rays_d, dirs], -1)

    rgbs = render(rays, model, random_sample)
    rgb_shape = list(origin_shape[:-1]) + list(rgbs.shape[1:])
    rgbs = torch.reshape(rgbs, rgb_shape)

    return rgbs

# render full image
def render_image(model, render_poses, hwf, outdir=None):
    for i, cam2world in enumerate(tqdm(render_poses)):
        rgb = render_rays(model, False, hwf, cam2world=cam2world[:3,:4])
        rgb8 = convert28bit(rgb.cpu().numpy())
        filename = os.path.join(outdir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)