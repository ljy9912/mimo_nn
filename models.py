'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cone_projection import soc, soc_leaky


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act_fun='ReLU', angle_tan=0.84):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if act_fun == 'ReLU':
            self.act_fun1 = nn.ReLU()
            self.act_fun2 = nn.ReLU()
        elif act_fun == 'LeakyReLU':
            self.act_fun1 = nn.LeakyReLU()
            self.act_fun2 = nn.LeakyReLU()
        elif act_fun == 'PReLU':
            self.act_fun1 = nn.PReLU()
            self.act_fun2 = nn.PReLU()
        elif act_fun == 'soc_2dim':
            self.act_fun1 = soc(angle_tan=angle_tan, cone_dim=2)
            self.act_fun2 = soc(angle_tan=angle_tan, cone_dim=2)
        elif act_fun == 'soc':
            self.act_fun1 = soc(angle_tan=angle_tan, cone_dim=3)
            self.act_fun2 = soc(angle_tan=angle_tan, cone_dim=3)
        elif act_fun == 'soc_2dim_leaky':
            self.act_fun1 = soc_leaky(angle_tan=angle_tan, cone_dim=2)
            self.act_fun2 = soc_leaky(angle_tan=angle_tan, cone_dim=2)
        else:
            raise ValueError('Activation not implemented! Please choose from ReLU, LeakyReLU, PReLU, soc_2dim, soc, soc_2dim_leaky!')

    def forward(self, x):
        out = self.act_fun1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act_fun2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, act_fun='ReLU', angle_tan=0.84):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if act_fun == 'ReLU':
            self.act_fun1 = nn.ReLU()
            self.act_fun2 = nn.ReLU()
            self.act_fun3 = nn.ReLU()
        elif act_fun == 'LeakyReLU':
            self.act_fun1 = nn.LeakyReLU()
            self.act_fun2 = nn.LeakyReLU()
            self.act_fun3 = nn.LeakyReLU()
        elif act_fun == 'PReLU':
            self.act_fun1 = nn.PReLU()
            self.act_fun2 = nn.PReLU()
            self.act_fun3 = nn.PReLU()
        elif act_fun == 'soc':
            self.act_fun1 = soc(angle_tan=angle_tan, cone_dim=3)
            self.act_fun2 = soc(angle_tan=angle_tan, cone_dim=3)
            self.act_fun3 = soc(angle_tan=angle_tan, cone_dim=3)
        elif act_fun == 'soc_2dim':
            self.act_fun1 = soc(angle_tan=angle_tan, cone_dim=2)
            self.act_fun2 = soc(angle_tan=angle_tan, cone_dim=2)
            self.act_fun3 = soc(angle_tan=angle_tan, cone_dim=2)
        elif act_fun == 'soc_2dim_leaky':
            self.act_fun1 = soc_leaky(angle_tan=angle_tan, cone_dim=2)
            self.act_fun2 = soc_leaky(angle_tan=angle_tan, cone_dim=2)
            self.act_fun3 = soc_leaky(angle_tan=angle_tan, cone_dim=2)

    def forward(self, x):
        out = self.act_fun1(self.bn1(self.conv1(x)))
        out = self.act_fun2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act_fun3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act_fun='ReLU', angle_tan=0.84):
        super(ResNet, self).__init__()
        self.act_fun_type = act_fun
        self.angle_tan = angle_tan
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if act_fun == 'ReLU':
            self.act_fun = nn.ReLU()
        elif act_fun == 'LeakyReLU':
            self.act_fun = nn.LeakyReLU()
        elif act_fun == 'PReLU':
            self.act_fun = nn.PReLU()
        elif act_fun == 'soc':
            self.act_fun = soc(angle_tan=angle_tan, cone_dim=3)
        elif act_fun == 'soc_2dim':
            self.act_fun = soc(angle_tan=angle_tan, cone_dim=2)
        elif act_fun == 'soc_2dim_leaky':
            self.act_fun = soc_leaky(angle_tan=angle_tan, cone_dim=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_fun=self.act_fun_type, angle_tan=self.angle_tan))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_fun(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(act_fun='ReLU', angle_tan=0.84):
    return ResNet(BasicBlock, [2, 2, 2, 2], act_fun=act_fun, angle_tan=angle_tan)


def ResNet34(act_fun='ReLU'):
    return ResNet(BasicBlock, [3, 4, 6, 3], act_fun=act_fun)


def ResNet50(act_fun='ReLU'):
    return ResNet(Bottleneck, [3, 4, 6, 3], act_fun=act_fun)


def ResNet101(act_fun='ReLU'):
    return ResNet(Bottleneck, [3, 4, 23, 3], act_fun=act_fun)


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 4, 32, 32))
    print(y.size())

# test()
