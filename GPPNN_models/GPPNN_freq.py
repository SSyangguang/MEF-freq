# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init
from GPPNN_models.utils.CDC import cdcconv


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU(0.1))

    def forward(self, x):
        out = self.conv(x)
        return out


class SpaBranch(nn.Module):
    def __init__(self, channels, depth=3):
        super(SpaBranch, self).__init__()
        self.depth = depth
        self.input = nn.Sequential(nn.Conv2d(channels, channels, (3, 3), (1, 1), 1),
                                   nn.Conv2d(channels, channels, (3, 3), (1, 1), 1),
                                   nn.LeakyReLU(0.1))
        self.conv = nn.ModuleList([
            nn.Sequential(ConvBlock(channels * (i + 1), channels)) for i in range(depth)
        ])

    def forward(self, x):
        x = self.input(x)
        for i in range(self.depth):
            t = self.conv[i](x)
            x = torch.cat([x, t], dim=1)

        return x


class FreBranch(nn.Module):
    def __init__(self, channels):
        super(FreBranch, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1), (1, 1), 0)

    def forward(self, x):
        freq = torch.fft.rfft2(self.conv(x) + 1e-8, norm='backward')
        amp = torch.abs(freq)
        pha = torch.angle(freq)

        return amp, pha


class FreTest(nn.Module):
    def __init__(self, channels):
        super(FreTest, self).__init__()
        self.fre_over = FreBranch(channels=channels)
        self.fre_under = FreBranch(channels=channels)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, (1, 1), (1, 1), 0),
                                      nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, (1, 1), (1, 1), 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, (1, 1), (1, 1), 0),
                                      nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, (1, 1), (1, 1), 0))
        self.fre_output = nn.Conv2d(channels, channels, (1, 1), (1, 1), 0)

    def forward(self, over, under):
        _, _, H, W = over.shape

        over_amp, over_pha = self.fre_over(over)
        under_amp, under_pha = self.fre_under(under)
        fre_amp = self.amp_fuse(torch.cat([over_amp, under_amp], dim=1))
        fre_pha = self.pha_fuse(torch.cat([over_pha, under_pha], dim=1))

        real = fre_amp * torch.cos(fre_pha) + 1e-8
        imag = fre_amp * torch.sin(fre_pha) + 1e-8

        fusion_fre = torch.complex(real, imag) + 1e-8
        fre_output = self.fre_output(torch.abs(torch.fft.irfft2(fusion_fre, s=(H, W), norm='backward')))

        return fre_output


class SpaTest(nn.Module):
    def __init__(self, channels, depth=3):
        super(SpaTest, self).__init__()
        self.depth = depth
        self.spa_over = SpaBranch(channels=2*channels, depth=3)

        self.spa_post = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(channels * (self.depth + 1) * 2, channels * 4, (3, 3), (1, 1), 1),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(channels * 4, channels * 2, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.1)
            ))
        self.spa_output = nn.Conv2d(channels, channels, (3, 3), (1, 1), 1)

    def forward(self, over, under):
        fusion_spa = torch.cat([over, under], dim=1)
        fusion_spa = self.spa_over(fusion_spa)
        fusion_spa = self.spa_post(fusion_spa)
        fusion_spa = self.spa_output(fusion_spa)

        return fusion_spa


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3





class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


class SpaFre(nn.Module):
    def __init__(self, channels):
        super(SpaFre, self).__init__()
        self.panprocess = nn.Conv2d(channels,channels,3,1,1)
        self.panpre = nn.Conv2d(channels,channels,1,1,0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        # self.spa_process = SpaTest(channels)
        self.fre_process = FreTest(channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, msf, pan):
        panpre = self.panprocess(pan)
        panf = self.panpre(panpre)
        # spafuse = self.spa_process(torch.cat([msf, panf], dim=1))
        frefuse = self.fre_process(msf,panf)
        # spa_map = self.spa_att(spafuse-frefuse)
        # spa_res = frefuse*spa_map+spafuse
        # cat_f = torch.cat([spa_res,frefuse],1)
        # cha_res = self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f))*cat_f)
        # out = cha_res+msf

        return frefuse, panpre


class FeatureProcess(nn.Module):
    def __init__(self, channels):
        super(FeatureProcess, self).__init__()

        self.conv_p = nn.Conv2d(1, channels, 3, 1, 1)
        self.conv_p1 = nn.Conv2d(1, channels, 3, 1, 1)
        self.block = SpaFre(channels)
        self.block1 = SpaFre(channels)
        self.block2 = SpaFre(channels)
        self.block3 = SpaFre(channels)
        self.block4 = SpaFre(channels)
        self.fuse = nn.Conv2d(channels,1,1,1,0)


    def forward(self, ms, pan):
        msf = self.conv_p(ms)
        panf = self.conv_p1(pan)
        msf0, panf0 = self.block(msf, panf)
        # msf1, panf1 = self.block1(msf0,panf0)
        # msf2, panf2 = self.block2(msf1, panf1)
        # msf3, panf3 = self.block3(msf2, panf2)
        # msf4, panf4 = self.block4(msf3, panf3)
        # msout = self.fuse(torch.cat([msf0,msf1,msf2,msf3,msf4],1))
        msout = self.fuse(msf0)

        return msout


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class GPPNN(nn.Module):
    def __init__(self, channels):
        super(GPPNN, self).__init__()
        # self.process = FeatureProcess(channels)
        self.process = FeatureProcess(channels)
        # self.cdc = nn.Sequential(nn.Conv2d(1, 4, 1, 1, 0), cdcconv(4, 4), cdcconv(4, 4))
        self.refine = Refine(channels, 1)

    def forward(self, ms, pan):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        # mHR = upsample(ms, M, N)
        HRf = self.process(ms, pan)
        # HR = self.refine(HRf) + ms
        # HR = self.refine(HRf)

        return HRf



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
