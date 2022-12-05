
import torch
import torch.nn as nn
import math

#https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
class SHEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # section 5.3 
        # degree = 4, coefficients = 16
        self.output_dim = 16

        self.l0 = 1./(2 * (math.pi ** 0.5))

        self.l1 = 0.5 * ((3./math.pi) ** 0.5)

        self.l2m_2 = 0.5 * ((15./math.pi) ** 0.5)
        self.l2m_1 = 0.5 * ((15./math.pi) ** 0.5)
        self.l2m0 = 0.25 * ((5./math.pi) ** 0.5)
        self.l2m1 = 0.5 * ((15./math.pi) ** 0.5)
        self.l2m2 = 0.25 * ((15./math.pi) ** 0.5)

        self.l3m_3 = 0.25 * ((17.5/math.pi) ** 0.5)
        self.l3m_2 = 0.5 * ((105./math.pi) ** 0.5)
        self.l3m_1 = 0.25 * ((10.5/math.pi) ** 0.5)
        self.l3m0 = 0.25 * ((7./math.pi) ** 0.5)
        self.l3m1 = 0.25 * ((10.5/math.pi) ** 0.5)
        self.l3m2 = 0.25 * ((105./math.pi) ** 0.5)
        self.l3m3 = 0.25 * ((17.5/math.pi) ** 0.5)


    def forward(self, input):

        output = torch.empty((*input.shape[:-1], self.output_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        output[..., 0] = self.l0

        output[..., 1] = -self.l1 * y
        output[..., 2] = self.l1 * z
        output[..., 3] = -self.l1 * x

        output[..., 4] = self.l2m_2 * x * y
        output[..., 5] = -self.l2m_1 * y * z
        output[..., 6] = self.l2m0 * (2.0 * z*z - x*x - y*y)
        output[..., 7] = -self.l2m1 * x * z
        output[..., 8] = self.l2m2 * (x*x - y*y)

        output[..., 9] = -self.l3m_3 * y * (3 * x*x - y*y)
        output[..., 10] = self.l3m_2 * x * y * z
        output[..., 11] = -self.l3m_1 * y * (4 * z*z - x*x - y*y)
        output[..., 12] = self.l3m0 * z * (2 * z*z - 3 * x*x - 3 * y*y)
        output[..., 13] = -self.l3m1 * x * (4 * z*z - x*x - y*y)
        output[..., 14] = self.l3m2 * z * (x*x - y*y)
        output[..., 15] = -self.l3m3 * x * (x*x - 3 * y*y)

        return output 