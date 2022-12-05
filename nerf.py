import torch
import torch.nn as nn
import torch.nn.functional as F

# section 4 implementation
class NeRF(nn.Module):
    def __init__(self, hash_encoder, sh_encoder):
        super(NeRF, self).__init__()

        self.hash_out_dim = hash_encoder.L * hash_encoder.F
        self.sh_out_dim = sh_encoder.output_dim
        self.num_density_layers = 2
        self.density_hidden_dim = 64
        self.num_color_layers = 4 
        self.hidden_dim_color = 64
        self.density_out_dim = 15
        self.hash_encoder = hash_encoder
        self.sh_encoder = sh_encoder


        density_MLP = []
        for layer in range(self.num_density_layers):
            if layer == 0:
                in_dim = self.hash_out_dim
            else:
                in_dim = self.density_hidden_dim
            
            if layer == self.num_density_layers - 1:
                out_dim = 1 + self.density_out_dim
            else:
                out_dim = self.density_hidden_dim
            
            density_MLP.append(nn.Linear(in_dim, out_dim, bias=False))

        self.density_MLP = nn.ModuleList(density_MLP)

        
        color_MLP =  []
        for layer in range(self.num_color_layers):
            if layer == 0:
                in_dim = self.sh_out_dim + self.density_out_dim
            else:
                in_dim = self.hidden_dim_color
            
            if layer == self.num_color_layers - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim_color
            
            color_MLP.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_MLP = nn.ModuleList(color_MLP)

    def get_parameters(self, lr, weight_decay):
        return [{'params': self.hash_encoder.parameters(), 'lr': lr},
                {'params': self.sh_encoder.parameters(), 'lr': lr},
                {'params': self.density_MLP.parameters(), 'weight_decay': weight_decay, 'lr': lr},
                {'params': self.color_MLP.parameters(), 'weight_decay': weight_decay, 'lr': lr}]
    
    def forward(self, x, d):
        i = self.hash_encoder(x)
        for layer in range(self.num_density_layers):
            i = self.density_MLP[layer](i)
            if layer != self.num_density_layers - 1:
                i = F.relu(i, inplace=True)

        sigma, density_out = i[...,0], i[...,1:]
        
        d = self.sh_encoder(d)
        i = torch.cat([d, density_out], dim=-1)
        for layer in range(self.num_color_layers):
            i = self.color_MLP[layer](i)
            if layer != self.num_color_layers - 1:
                i = F.relu(i, inplace=True)

        color = i
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs