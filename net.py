import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from option import args
from GPPNN_models.refine import Refine
from GPPNN_models.modules import InvertibleConv1x1
from GPPNN_models.GPPNN import Freprocess

from collections import OrderedDict


# network functions
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type='prelu', slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(negative_slope=slope, inplace=True)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % norm_type)
    return layer


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return nn.Sequential(conv, n, act)


class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(DenseLayer, self).__init__()
        self.conv = ConvBlock(num_channels, growth, 3, act_type='lrelu', norm_type=None)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, num_features):
        super(DenseNet, self).__init__()
        self.num_channels = 1
        self.num_features = num_features
        self.growth = 44
        modules = []
        self.conv_1 = ConvBlock(2 * self.num_channels, self.num_features, kernel_size=3, act_type='lrelu', norm_type=None)
        for i in range(5):
            modules.append(DenseLayer(self.num_features, self.growth))
            self.num_features += self.growth
        self.dense_layers = nn.Sequential(*modules)
        self.sub = nn.Sequential(ConvBlock(self.num_features, 128, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
                                 nn.Conv2d(32, num_features, 3, 1, 1),
                                 )

    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.sub(x)
        return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


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
    def __init__(self, channel_in, channel_out, d=1, init='xavier', gc=8, bias=True):
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


# class InvBlock(nn.Module):
#     def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
#         super(InvBlock, self).__init__()
#         # channel_num: 3
#         # channel_split_num: 1
#
#         self.split_len1 = channel_split_num  # 1
#         self.split_len2 = channel_num - channel_split_num  # 2
#
#         self.clamp = clamp
#
#         self.F = subnet_constructor(self.split_len2, self.split_len1, d)
#         self.G = subnet_constructor(self.split_len1, self.split_len2, d)
#         self.H = subnet_constructor(self.split_len1, self.split_len2, d)
#
#         in_channels = channel_num
#         self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
#         self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
#
#     def forward(self, x, rev=False):
#         # if not rev:
#         # invert1x1conv
#         x, logdet = self.flow_permutation(x, logdet=0, rev=False)
#
#         # split to 1 channel and 2 channel.
#         x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
#
#         y1 = x1 + self.F(x2)  # 1 channel
#         self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
#         y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
#         out = torch.cat((y1, y2), 1)
#
#         return out


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1),
#                                   nn.BatchNorm2d(out_channels),
#                                   nn.LeakyReLU(0.1))
#
#     def forward(self, x):
#         out = self.conv(x)
#         return out


# class SpaBranch(nn.Module):
#     def __init__(self, channels, depth=3):
#         super(SpaBranch, self).__init__()
#         self.depth = depth
#         self.input = nn.Sequential(nn.Conv2d(channels, channels, (3, 3), (1, 1), 1),
#                                    nn.Conv2d(channels, channels, (3, 3), (1, 1), 1),
#                                    nn.LeakyReLU(0.1))
#         self.conv = nn.ModuleList([
#             nn.Sequential(ConvBlock(channels * (i + 1), channels)) for i in range(depth)
#         ])
#
#     def forward(self, x):
#         x = self.input(x)
#         for i in range(self.depth):
#             t = self.conv[i](x)
#             x = torch.cat([x, t], dim=1)
#
#         return x


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


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class Fusion(nn.Module):
    def __init__(self, channels, depth=3):
        super(Fusion, self).__init__()
        self.depth = depth
        self.channels = channels
        self.dense = DenseNet(self.channels)
        self.over_input = nn.Sequential(nn.Conv2d(1, self.channels, (3, 3), (1, 1), 1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1), 0),
                                        nn.LeakyReLU(0.1)
                                        )
        self.under_input = nn.Sequential(nn.Conv2d(1, self.channels, (3, 3), (1, 1), 1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1), 0),
                                         nn.LeakyReLU(0.1)
                                         )

        self.fref0 = FreTest(channels=self.channels)
        self.fref1 = FreTest(channels=self.channels)
        self.fref2 = FreTest(channels=self.channels)
        self.fref3 = FreTest(channels=self.channels)

        self.spa_post = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(self.channels * (self.depth + 1) * 2, self.channels * 4, (3, 3), (1, 1), 1),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(self.channels * 4, self.channels * 2, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(self.channels * 2),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(self.channels * 2, self.channels, (3, 3), (1, 1), 1),
                nn.BatchNorm2d(self.channels),
                nn.LeakyReLU(0.1)
            ))
        self.spa_output = nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1)
        self.fre_output = nn.Sequential(nn.Conv2d(self.channels*4, self.channels*2, (1, 1), (1, 1), 0),
                                        nn.Conv2d(self.channels*2, self.channels, (1, 1), (1, 1), 0))

        self.spa_att = nn.Sequential(nn.Conv2d(self.channels, self.channels // 2, (3, 3), (1, 1), 1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(self.channels // 2, self.channels, (3, 3), (1, 1), 1, bias=True),
                                     nn.Sigmoid())

        self.output = nn.Sequential(
            nn.Conv2d(self.channels, 4, (3, 3), (1, 1), 1),
            nn.Conv2d(4, 1, (3, 3), (1, 1), 1),
            nn.Tanh()
        )

        self.refine = Refine(self.channels, 1)

        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, (3, 3), (1, 1), 1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, (3, 3), (1, 1), 1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, (1, 1), (1, 1), 0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, (1, 1), (1, 1), 0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, (3, 3), (1, 1), 1, bias=True)
        self.fre_loss = nn.Conv2d(self.channels, 1, (1, 1), (1, 1), 0, bias=True)


    def forward(self, over, under):
        _, _, H, W = over.shape

        spa_output = self.dense(over, under)

        over = self.over_input(over)
        under = self.under_input(under)

        fre_f0 = self.fref0(over, under)
        fre_f1 = self.fref1(over, under)
        fre_f2 = self.fref2(over, under)
        fre_f3 = self.fref3(over, under)
        fre_output = self.fre_output(torch.cat([fre_f0, fre_f1, fre_f2, fre_f3], dim=1))

        spa_map = self.spa_att(spa_output - fre_output)
        spa_res = fre_output * spa_map + spa_output
        cat_f = torch.cat([spa_res, fre_output], 1)
        fusion = self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f)
        fusion = self.output(fusion)

        fre_loss = self.fre_loss(fre_output)

        # fusion_spa = self.spa_post(fusion_spa)
        # spa_output = self.spa_output(fusion_spa)
        # spa_fre = torch.fft.rfft2(spa_output + 1e-8, norm='backward')
        # spa_amp = torch.abs(spa_fre)
        # spa_pha = torch.angle(spa_fre)
        #
        # fre_fre = torch.fft.rfft2(fre_output + 1e-8, norm='backward')
        # fre_amp = torch.abs(fre_fre)
        # fre_pha = torch.angle(fre_fre)
        #
        # fusion_fre_amp = fre_amp + spa_amp
        # pha_map = self.spa_att(spa_pha - fre_pha)
        # fusion_fre_pha = spa_pha * pha_map + spa_pha
        #
        # real = fusion_fre_amp * torch.cos(fusion_fre_pha) + 1e-8
        # imag = fusion_fre_amp * torch.sin(fusion_fre_pha) + 1e-8
        # fusion_fre = torch.complex(real, imag) + 1e-8
        # fusion = self.output(torch.abs(torch.fft.irfft2(fusion_fre, s=(H, W), norm='backward')))

        # fusion = self.output(spa_output + fre_output)

        # fusion = self.output(torch.cat([spa_output, fre_output], 1))

        # fre_output = self.refine(fre_output)

        return fusion, fre_loss


# class Fusion(nn.Module):
#     def __init__(self, channels, depth=3):
#         super(Fusion, self).__init__()
#         self.depth = depth
#         self.over_input = nn.Conv2d(1, channels, (3, 3), (1, 1), 1)
#         self.under_input = nn.Conv2d(1, channels, (3, 3), (1, 1), 1)
#
#         # self.spa_over = SpaBranch(channels=channels, depth=3)
#         # self.spa_under = SpaBranch(channels=channels, depth=3)
#         # self.fre_over = FreBranch(channels=channels)
#         # self.fre_under = FreBranch(channels=channels)
#         self.spa_over = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
#                                          nn.Conv2d(2*channels,channels,1,1,0))
#         self.spa_under = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
#                                          nn.Conv2d(2*channels,channels,1,1,0))
#         self.fre_over = Freprocess(channels=channels)
#         self.fre_under = Freprocess(channels=channels)
#
#         self.amp_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, (1, 1), (1, 1), 0),
#                                       nn.LeakyReLU(0.1, inplace=False),
#                                       nn.Conv2d(channels, channels, (1, 1), (1, 1), 0))
#         self.pha_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, (1, 1), (1, 1), 0),
#                                       nn.LeakyReLU(0.1, inplace=False),
#                                       nn.Conv2d(channels, channels, (1, 1), (1, 1), 0))
#
#         self.spa_post = nn.Sequential(
#             nn.Sequential(
#                 nn.Conv2d(channels * (self.depth + 1) * 2, channels * 4, (3, 3), (1, 1), 1),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv2d(channels * 4, channels * 2, (3, 3), (1, 1), 1),
#                 nn.BatchNorm2d(channels * 2),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv2d(channels * 2, channels, (3, 3), (1, 1), 1),
#                 nn.BatchNorm2d(channels),
#                 nn.LeakyReLU(0.1)
#             ))
#         self.spa_output = nn.Conv2d(channels, channels, (3, 3), (1, 1), 1)
#         self.fre_output = nn.Conv2d(channels, channels, (1, 1), (1, 1), 0)
#
#         self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, (3, 3), (1, 1), 1, bias=True),
#                                      nn.LeakyReLU(0.1),
#                                      nn.Conv2d(channels // 2, channels, (3, 3), (1, 1), 1, bias=True),
#                                      nn.Sigmoid())
#
#         self.output = nn.Sequential(
#             nn.Conv2d(channels * 2, 1, (3, 3), (1, 1), 1),
#             nn.Tanh()
#         )
#
#         self.refine = Refine(channels, 3)
#
#     def forward(self, over, under):
#         _, _, H, W = over.shape
#
#         over = self.over_input(over)
#         under = self.under_input(under)
#
#         fusion_spa = self.spa_over(torch.cat([over, under], dim=1))
#         # under_spa = self.spa_under(under)
#         fusion_fre = self.fre_over(over, under)
#         # under_amp, under_pha = self.fre_under(under)
#
#         # fusion_spa = torch.cat([over_spa, under_spa], dim=1)
#         # fre_mag = self.amp_fuse(torch.cat([over_amp, under_amp], dim=1))
#         # fre_pha = self.pha_fuse(torch.cat([over_pha, under_pha], dim=1))
#
#         # fusion_spa = self.spa_post(fusion_spa)
#         # spa_output = self.spa_output(fusion_spa)
#         # spa_fre = torch.fft.rfft2(fusion_spa + 1e-8, norm='backward')
#         # spa_amp = torch.abs(spa_fre)
#         # spa_pha = torch.angle(spa_fre)
#
#         # fusion_fre_amp = fre_mag + spa_amp
#         # pha_map = self.spa_att(spa_pha - fre_pha)
#         # fusion_fre_pha = spa_pha * pha_map + spa_pha
#
#         # real = fusion_fre_amp * torch.cos(fusion_fre_pha) + 1e-8
#         # imag = fusion_fre_amp * torch.sin(fusion_fre_pha) + 1e-8
#
#         # real = fre_mag * torch.cos(fre_pha) + 1e-8
#         # imag = fre_mag * torch.sin(fre_pha) + 1e-8
#         #
#         # fusion_fre = torch.complex(real, imag) + 1e-8
#         # fre_output = self.fre_output(torch.abs(torch.fft.irfft2(fusion_fre, s=(H, W), norm='backward')))
#
#         # fusion = self.output(spa_output + fre_output)
#         fusion = self.output(torch.cat([fusion_spa, fusion_fre], 1))
#
#         # fusion = self.refine(fusion)
#
#         return fusion