import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def histcal(x, bins=256, min=0.0, max=1.0):
    n,c,h,w = x.size()
    n_batch = n
    row_m = h
    row_n = w
    channels = c

    delta = (max - min) / bins
    BIN_Table = np.arange(0, bins, 1)
    BIN_Table = BIN_Table * delta

    zero = torch.tensor([[[0.0]]],requires_grad=False).cuda()
    zero = zero.repeat(n,c,1)
    temp = torch.ones(size=x.size()).cuda()
    temp1 = torch.zeros(size=x.size()).cuda()
    for dim in range(1, bins - 1, 1):
        h_r = BIN_Table[dim]  # h_r
        h_r_sub_1 = BIN_Table[dim - 1]  # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1]  # h_(r+1)

        h_r = torch.tensor(h_r).float().cuda()
        h_r_sub_1 = torch.tensor(h_r_sub_1).float().cuda()
        h_r_plus_1 = torch.tensor(h_r_plus_1).float().cuda()

        h_r_temp = h_r * temp
        h_r_sub_1_temp = h_r_sub_1 * temp
        h_r_plus_1_temp = h_r_plus_1 * temp

        mask_sub = torch.where(torch.greater(h_r_temp, x) & torch.greater(x, h_r_sub_1_temp), temp, temp1)
        mask_plus = torch.where(torch.greater(x, h_r_temp) & torch.greater(h_r_plus_1_temp, x), temp, temp1)

        temp_mean1 = torch.mean((((x - h_r_sub_1) * mask_sub).view(n_batch, channels, -1)), dim=-1)
        temp_mean2 = torch.mean((((h_r_plus_1 - x) * mask_plus).view(n_batch, channels, -1)), dim=-1)

        if dim == 1:
            temp_mean = torch.add(temp_mean1, temp_mean2)
            temp_mean = torch.unsqueeze(temp_mean, -1)  # [1,1,1]
        else:
            if dim != bins - 2:
                temp_mean_temp = torch.add(temp_mean1, temp_mean2)
                temp_mean_temp = torch.unsqueeze(temp_mean_temp, -1)
                temp_mean = torch.cat([temp_mean, temp_mean_temp], dim=-1)
            else:
                zero = torch.cat([zero, temp_mean], dim=-1)
                temp_mean_temp = torch.add(temp_mean1, temp_mean2)
                temp_mean_temp = torch.unsqueeze(temp_mean_temp, -1)
                temp_mean = torch.cat([temp_mean, temp_mean_temp], dim=-1)

    # diff = torch.abs(temp_mean - zero)
    return temp_mean
