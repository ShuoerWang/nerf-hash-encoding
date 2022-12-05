import torch
import os
import argparse
import warnings

from load_blender import load_blender_data
from render import render_image

from nerf import NeRF
from hash_encoding import HashEncoder
from sh_encoding import SHEncoder
from trainer import Trainer

# Section 4 training
learning_rate = 0.01
L2_regularization = 1e-6
batch_size = 2 ** 10

device = torch.device("cuda")
    

if __name__=='__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='--data: relative path to data\n--iter: number of iterations')
    parser.add_argument('--data', type=str, nargs='?', help='input path', const = "./data/chair", default="./data/chair")
    parser.add_argument('--iter', type=int, nargs='?', help='number of ', const = 1000, default=1000)
    args = parser.parse_args()

    # avoid "found two devices: cuda:0 and cpu" error
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # load data
    images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.data, True, 8)
    images_idx, i_val, i_test = i_split

    # back ground color needs to be white
    images = images[...,:3] * images[...,3:] + (1-images[...,3:])

    # photo info
    height, width, focal = hwf
    height, width = int(height), int(width)
    hwf = [height, width, focal]

    # create nerf encoders, model and start to train
    hash_encodor = HashEncoder(bounding_box)
    sh_encoder = SHEncoder()
    model = NeRF(hash_encodor, sh_encoder).to(device)

    trainer = Trainer(model, images, images_idx, poses, hwf, learning_rate, L2_regularization, batch_size)
    trainer.train(args.iter)
   
    os.makedirs('./out/', exist_ok=True)
    with torch.no_grad():
        render_image(model, torch.Tensor(poses[i_test]).to(device), hwf, outdir='./out/')
