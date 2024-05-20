import torch
import numpy as np
import torch.nn as nn

class soc_with_angle(nn.Module):
    def __init__(self, angle_tan=1.19, cone_dim=2):
        '''
        angle_tan: the tan value of the cone's half-apex angle (rad)
        cone_dim: the dimension of the cone projection. We usually choose 2 or 3.
        '''
        super().__init__()
        self.cone_dim = cone_dim
        self.angle_tan = angle_tan

    def forward(self, x):
        '''
        x: input data, batch_size * (width * height) * feature_dimension. Please make sure that the feature dimension is at the last dimension.
        '''
        x_dim = len(x.shape)
        if x.size(x_dim-1) % self.cone_dim == 0:
            x_reshape = torch.clone(x)
        else:
            for dim in range(1, self.cone_dim):
                if x.size(x_dim-1) % self.cone_dim == dim:
                    x_reshape = torch.cat([x, torch.zeros(tuple(x.shape[0:-1]) + (self.cone_dim-dim,)).cuda()], dim=x_dim-1)

        x_reshape = x_reshape.contiguous().view(tuple([x.size(i) for i in range(x_dim-1)]) + (int(x_reshape.size(x_dim-1) / self.cone_dim), self.cone_dim))

        s = 1 / np.sqrt(self.cone_dim) * torch.sum(x_reshape, dim=x_dim)
        y = x_reshape - s.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) / np.sqrt(self.cone_dim)

        angle_tan = self.angle_tan
        y_norm = torch.linalg.norm(y, 2, dim=x_dim)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) * y

        x_result = s1.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) / np.sqrt(self.cone_dim) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(x_dim-1) % self.cone_dim == 0:
            return x_result2.contiguous().view(tuple([x.size(i) for i in range(x_dim)]))
        else:
            for dim in range(1, x_dim):
                if x.size(x_dim-1) % self.cone_dim == dim:
                    return x_result2.contiguous().view(tuple([x.size(i) for i in range(x_dim-1)]) + (x.size(x_dim-1)+self.cone_dim-dim,))[..., :-(self.cone_dim-dim)]

class soc(nn.Module):
    def __init__(self, angle_tan=1.19, cone_dim=2):
        '''
        angle_tan: the tan value of the cone's half-apex angle (rad)
        cone_dim: the dimension of the cone projection. We usually choose 2 or 3.
        '''
        super().__init__()
        self.cone_dim = cone_dim
        self.angle_tan = nn.Parameter(torch.tensor([0.0])).cuda()
        self.angle_tan_init = angle_tan

    def forward(self, x):
        '''
        x: input data, batch_size * (width * height) * feature_dimension. Please make sure that the feature dimension is at the last dimension.
        '''
        x_dim = len(x.shape)
        if x.size(x_dim-1) % self.cone_dim == 0:
            x_reshape = torch.clone(x)
        else:
            for dim in range(1, self.cone_dim):
                if x.size(x_dim-1) % self.cone_dim == dim:
                    x_reshape = torch.cat([x, torch.zeros(tuple(x.shape[0:-1]) + (self.cone_dim-dim,)).cuda()], dim=x_dim-1)

        x_reshape = x_reshape.contiguous().view(tuple([x.size(i) for i in range(x_dim-1)]) + (int(x_reshape.size(x_dim-1) / self.cone_dim), self.cone_dim))

        s = 1 / np.sqrt(self.cone_dim) * torch.sum(x_reshape, dim=x_dim)
        y = x_reshape - s.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) / np.sqrt(self.cone_dim)

        angle_tan = self.angle_tan
        y_norm = torch.linalg.norm(y, 2, dim=x_dim)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) * y

        x_result = s1.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) / np.sqrt(self.cone_dim) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(x_dim-1) % self.cone_dim == 0:
            return x_result2.contiguous().view(tuple([x.size(i) for i in range(x_dim)]))
        else:
            for dim in range(1, x_dim):
                if x.size(x_dim-1) % self.cone_dim == dim:
                    return x_result2.contiguous().view(tuple([x.size(i) for i in range(x_dim-1)]) + (x.size(x_dim-1)+self.cone_dim-dim,))[..., :-(self.cone_dim-dim)]

class soc_leaky(nn.Module):
    def __init__(self, angle_tan=1.19, cone_dim=2):
        '''
        angle_tan: the tan value of the cone's half-apex angle (rad)
        cone_dim: the dimension of the cone projection. We usually choose 2 or 3.
        '''
        super().__init__()
        self.cone_dim = cone_dim
        self.angle_tan = nn.Parameter(torch.tensor([0.0]))
        self.angle_tan_init = angle_tan

    def forward(self, x):
        '''
        x: input data, batch_size * (width * height) * feature_dimension. Please make sure that the feature dimension is at the last dimension.
        '''
        x_dim = len(x.shape)
        if x.size(x_dim-1) % self.cone_dim == 0:
            x_reshape = torch.clone(x)
        else:
            for dim in range(1, self.cone_dim):
                if x.size(x_dim-1) % self.cone_dim == dim:
                    x_reshape = torch.cat([x, torch.zeros(tuple(x.shape[0:-1]) + (self.cone_dim-dim,)).cuda()], dim=x_dim-1)

        x_reshape = x_reshape.contiguous().view(tuple([x.size(i) for i in range(x_dim-1)]) + (int(x_reshape.size(x_dim-1) / self.cone_dim), self.cone_dim))

        s = 1 / np.sqrt(self.cone_dim) * torch.sum(x_reshape, dim=x_dim)
        y = x_reshape - s.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) / np.sqrt(self.cone_dim)

        angle_tan = self.angle_tan
        y_norm = torch.linalg.norm(y, 2, dim=x_dim)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) * y

        x_result = s1.unsqueeze(x_dim).repeat_interleave(self.cone_dim, dim=x_dim) / np.sqrt(self.cone_dim) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(x_dim-1) % self.cone_dim == 0:
            return x_result2.contiguous().view(tuple([x.size(i) for i in range(x_dim)])) * 0.99 + 0.01 * x
        else:
            for dim in range(1, x_dim):
                if x.size(x_dim-1) % self.cone_dim == dim:
                    return x_result2.contiguous().view(tuple([x.size(i) for i in range(x_dim-1)]) + (x.size(x_dim-1)+self.cone_dim-dim,))[..., :-(self.cone_dim-dim)] * 0.99 + 0.01 * x