import numpy as np
import torch
from render import get_rays, render_rays

device = torch.device("cuda")

class Trainer(object):
    def __init__(self, model, images, images_idxs, poses, hwf, lr, L2r, batch_size):
        self.batch_size = batch_size
        self.poses = poses
        self.images_idxs = images_idxs
        self.hwf = hwf
        self.height, self.width, self.focal = hwf
        self.model = model
        self.images = images
        
        # Section 4 training
        self.optimizer = torch.optim.Adam(self.model.get_parameters(lr, L2r), betas=(0.9, 0.99), eps=1e-15)

    # Section 4 training
    def L2_loss(self, result, golden):
        return torch.mean((result - golden) ** 2)

    def train(self, iteration):
        for i in range(iteration):
            # get all pixels and randomly select batch size of them
            height_vector = torch.linspace(0, self.height -1 , self.height)
            width_vector = torch.linspace(0, self.width - 1, self.width)

            pixels = torch.stack(torch.meshgrid(height_vector, width_vector), -1)
            pixels = torch.reshape(pixels, [-1,2])
            select_idx = np.random.choice(pixels.shape[0], size=[self.batch_size], replace=False)
            selection = pixels[select_idx].long()

            # select random image and get the photo info
            images_idx = np.random.choice(self.images_idxs)
            pose = self.poses[images_idx, :3,:4]

            # generate all rays and select them by the selected pixels
            rays_o, rays_d = get_rays(self.hwf, torch.Tensor(pose))
            rays_o = rays_o[selection[:, 0], selection[:, 1]]
            rays_d = rays_d[selection[:, 0], selection[:, 1]]

            # get the ground truth rgb values
            golden = self.images[images_idx]
            batch = torch.stack([rays_o, rays_d], 0)
            golden = torch.Tensor(golden).to(device)
            goldens = golden[selection[:, 0], selection[:, 1]]

            # train step
            rgb = render_rays(self.model, True, self.hwf, rays=batch)
            self.optimizer.zero_grad()
            img_loss = self.L2_loss(rgb, goldens)
            loss = img_loss
            loss.backward()
            self.optimizer.step()

            if (i % 100 == 0):
                print("loss: ", loss.cpu().detach().numpy(), "iteration: ", i)
