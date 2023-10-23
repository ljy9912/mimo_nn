import torch
import numpy as np
import torch.nn as nn

class soc_2dim_with_angle(nn.Module):
    def __init__(self, angle):
        '''
        angle_tan: the tan value of the cone's half-apex angle (rad)
        '''
        super().__init__()
        self.input_dimension = 2
        self.angle_tan = np.tan(angle)

    def forward(self, x):
        '''
        x: input data, batch_size * feature_dimension
        '''
        if x.size(1) % self.input_dimension == 1:
            x_reshape = torch.cat([x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        else:
            x_reshape = torch.clone(x)
        x_reshape = x_reshape.contiguous().view(x_reshape.size(0), int(x_reshape.size(1) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=2)
        y = x_reshape - s.unsqueeze(self.input_dimension).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        angle_tan = self.angle_tan + 1.19
        y_norm = torch.linalg.norm(y, self.input_dimension, dim=2)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(2).repeat(1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(2).repeat(1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(2).repeat(1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(1) % self.input_dimension == 1:
            return x_result2.contiguous().view(x.size(0), x.size(1)+1)[:, :-1]
        else:
            return x_result2.contiguous().view(x.size(0), x.size(1))

class soc_2dim(nn.Module):
    def __init__(self):
        '''
        angle_tan: the tan value of the cone's half-apex angle (rad)
        '''
        super().__init__()
        self.input_dimension = 2
        self.angle_tan = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        '''
        x: input data, batch_size * feature_dimension
        '''
        if x.size(1) % self.input_dimension == 1:
            x_reshape = torch.cat([x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        else:
            x_reshape = torch.clone(x)
        x_reshape = x_reshape.contiguous().view(x_reshape.size(0), int(x_reshape.size(1) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=2)
        y = x_reshape - s.unsqueeze(self.input_dimension).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        angle_tan = self.angle_tan + 1.19
        y_norm = torch.linalg.norm(y, self.input_dimension, dim=2)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(2).repeat(1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(2).repeat(1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(2).repeat(1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(1) % self.input_dimension == 1:
            return x_result2.contiguous().view(x.size(0), x.size(1)+1)[:, :-1]
        else:
            return x_result2.contiguous().view(x.size(0), x.size(1))

class soc_2dim_leaky(nn.Module):
    def __init__(self):
        '''
        angle: the cone angle (rad)
        '''
        super().__init__()
        self.input_dimension = 2
        self.angle_tan = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        if x.size(1) % self.input_dimension == 1:
            x_reshape = torch.cat([x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        else:
            x_reshape = torch.clone(x)
        x_reshape = x_reshape.contiguous().view(x_reshape.size(0), int(x_reshape.size(1) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=2)
        y = x_reshape - s.unsqueeze(self.input_dimension).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        angle_tan = self.angle_tan + 1.19
        y_norm = torch.linalg.norm(y, self.input_dimension, dim=2)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(2).repeat(1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(2).repeat(1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(2).repeat(1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(1) % self.input_dimension == 1:
            return x_result2.contiguous().view(x.size(0), x.size(1)+1)[:, :-1] * 0.99 + 0.01 * x
        else:
            return x_result2.contiguous().view(x.size(0), x.size(1)) * 0.99 + 0.01 * x

class soc_3dim(nn.Module):
    def __init__(self):
        '''
        angle: the cone angle (rad)
        '''
        super().__init__()
        self.input_dimension = 3
        self.angle_tan = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        if x.size(1) % self.input_dimension == 1:
            x_reshape = torch.cat([torch.zeros(x.size(0), 1), x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        elif x.size(1) % self.input_dimension == 2:
            x_reshape = torch.cat([x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        else:
            x_reshape = torch.clone(x)
        x_reshape = x_reshape.contiguous().view(x_reshape.size(0), int(x_reshape.size(1) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=2)
        y = x_reshape - s.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        angle_tan = self.angle_tan + 1.19
        y_norm = torch.linalg.norm(y, self.input_dimension, dim=2)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(2).repeat(1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(2).repeat(1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(2).repeat(1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(1) % self.input_dimension == 1:
            return x_result2.contiguous().view(x.size(0), x.size(1)+2)[:, 1:-1]
        elif x.size(1) % self.input_dimension == 2:
            return x_result2.contiguous().view(x.size(0), x.size(1)+1)[:, :-1]
        else:
            return x_result2.contiguous().view(x.size(0), x.size(1))

class soc_3dim_leaky(nn.Module):
    def __init__(self):
        '''
        angle: the cone angle (rad)
        '''
        super().__init__()
        self.input_dimension = 3
        self.angle_tan = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        if x.size(1) % self.input_dimension == 1:
            x_reshape = torch.cat([torch.zeros(x.size(0), 1), x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        elif x.size(1) % self.input_dimension == 2:
            x_reshape = torch.cat([x, torch.zeros(x.size(0), 1).cuda()], dim=1)
        else:
            x_reshape = torch.clone(x)
        x_reshape = x_reshape.contiguous().view(x_reshape.size(0), int(x_reshape.size(1) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=2)
        y = x_reshape - s.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        angle_tan = self.angle_tan + 1.19
        y_norm = torch.linalg.norm(y, self.input_dimension, dim=2)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(2).repeat(1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(2).repeat(1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(2).repeat(1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(2).repeat(1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(1) % self.input_dimension == 1:
            return x_result2.contiguous().view(x.size(0), x.size(1)+2)[:, 1:-1] * 0.99 + 0.01 * x
        elif x.size(1) % self.input_dimension == 2:
            return x_result2.contiguous().view(x.size(0), x.size(1)+1)[:, :-1] * 0.99 + 0.01 * x
        else:
            return x_result2.contiguous().view(x.size(0), x.size(1)) * 0.99 + 0.01 * x

class soc_2dim_resnet(nn.Module):
    def __init__(self, angle_tan):
        '''
        angle: the cone angle (rad)
        '''
        super().__init__()
        self.input_dimension = 2
        self.angle_tan = nn.Parameter(torch.ones(1) * angle_tan)

    def forward(self, x):
        x_reshape = x.contiguous().view(x.size(0), x.size(1), x.size(2), int(x.size(3) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=4)
        y = x_reshape - s.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        self.angle_tan.data = torch.clamp(self.angle_tan.data, min=0.58, max=1.19)
        angle_tan = self.angle_tan
        y_norm = torch.linalg.norm(y, 2, dim=4)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        return x_result2.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(3))

class soc_2dim_leaky_resnet(nn.Module):
    def __init__(self, angle_tan):
        '''
        angle: the cone angle (rad)
        '''
        super().__init__()
        self.input_dimension = 2
        self.angle_tan = nn.Parameter(torch.zeros(1))
        self.angle_tan_value = angle_tan

    def forward(self, x):
        x_reshape = x.contiguous().view(x.size(0), x.size(1), x.size(2), int(x.size(3) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=4)
        y = x_reshape - s.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        angle_tan = self.angle_tan + self.angle_tan_value
        y_norm = torch.linalg.norm(y, 2, dim=4)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        return x_result2.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(3)) * 0.99 + 0.01 * x

class soc_resnet(nn.Module):
    def __init__(self, angle_tan):
        '''
        angle: the cone angle (rad)
        '''
        super().__init__()
        self.input_dimension = 3
        self.angle_tan = nn.Parameter(torch.tensor([angle_tan]))

    def forward(self, x):
        if x.size(3) % self.input_dimension == 1:
            x_reshape = torch.cat([torch.zeros(x.size(0), x.size(1), x.size(2), 1, device='cuda'), x, torch.zeros(x.size(0), x.size(1), x.size(2), 1, device='cuda')], dim=3)
        elif x.size(3) % self.input_dimension == 2:
            x_reshape = torch.cat([x, torch.zeros(x.size(0), x.size(1), x.size(2), 1, device='cuda')], dim=3)

        angle_tan = self.angle_tan
        x_reshape = x_reshape.contiguous().view(x_reshape.size(0), x_reshape.size(1), x_reshape.size(2), int(x_reshape.size(3) / self.input_dimension), self.input_dimension)

        s = 1 / np.sqrt(self.input_dimension) * torch.sum(x_reshape, dim=4)
        y = x_reshape - s.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) / np.sqrt(self.input_dimension)

        # angle_tan = np.tan(self.angle)
        y_norm = torch.linalg.norm(y, 2, dim=4)
        mask1 = torch.logical_and((s / angle_tan > -y_norm), (s * angle_tan < y_norm)).unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension)
        mask2 = (y_norm <= s * angle_tan).unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension)
        s1 = (s + y_norm * angle_tan) / (angle_tan ** 2 + 1)
        coeff = s1 / (y_norm + 1e-9) * angle_tan
        y1 = coeff.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) * y

        x_result = s1.unsqueeze(4).repeat(1, 1, 1, 1, self.input_dimension) / np.sqrt(self.input_dimension) + y1
        x_result2 = mask1 * x_result + mask2 * x_reshape

        if x.size(3) % self.input_dimension == 0:
            return x_result2.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(3))
        elif x.size(3) % self.input_dimension == 1:
            return x_result2.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(3)+2)[:, :, :, 1:-1]
        else:
            return x_result2.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(3)+1)[:, :, :, :-1]