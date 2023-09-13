import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


from option import args

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


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


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
    return sequential(conv, n, act)

# def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
#               act_type='prelu', norm_type='bn'):
#     if valid_padding:
#         padding = get_valid_padding(kernel_size, dilation)
#     else:
#         pass
#     conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
#                      bias=bias)
#
#     act = activation(act_type) if act_type else None
#     n = norm(out_channels, norm_type) if norm_type else None
#     return nn.Sequential(conv, n, act)


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
        self.growth = 44   # 之前是44
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


# def initialize_weights(net_l, scale=1):
#     if not isinstance(net_l, list):
#         net_l = [net_l]
#     for net in net_l:
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 m.weight.data *= scale  # for residual block
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias.data, 0.0)
#
#
# def initialize_weights_xavier(net_l, scale=1):
#     if not isinstance(net_l, list):
#         net_l = [net_l]
#     for net in net_l:
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)
#                 m.weight.data *= scale  # for residual block
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias.data, 0.0)
#
#
# class UNetConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, d, relu_slope=0.1):
#         super(UNetConvBlock, self).__init__()
#         self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
#
#         self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
#         self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
#         self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
#
#     def forward(self, x):
#         out = self.relu_1(self.conv_1(x))
#         out = self.relu_2(self.conv_2(out))
#         out += self.identity(x)
#
#         return out
#
#
# class DenseBlock(nn.Module):
#     def __init__(self, channel_in, channel_out, d=1, init='xavier', gc=8, bias=True):
#         super(DenseBlock, self).__init__()
#         self.conv1 = UNetConvBlock(channel_in, gc, d)
#         self.conv2 = UNetConvBlock(gc, gc, d)
#         self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         if init == 'xavier':
#             initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
#         else:
#             initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
#         # initialize_weights(self.conv5, 0)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(x1))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#
#         return x3


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


class FreBlock(nn.Module):
    def __init__(self, channels):
        super(FreBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (1, 1), (1, 1), 0)

    def forward(self, x):
        freq = torch.fft.rfft2(self.conv(x) + 1e-8, norm='backward')
        amp = torch.abs(freq)
        pha = torch.angle(freq)

        return amp, pha


class FreLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreLayer, self).__init__()
        self.fre_over = FreBlock(channels=in_channels)
        self.fre_under = FreBlock(channels=in_channels)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2 * in_channels, in_channels, (1, 1), (1, 1), 0),
                                      nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2 * in_channels, in_channels, (1, 1), (1, 1), 0),
                                      nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), 0))
        self.fre_output = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), 0)

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


class ChannelAtt(nn.Module):
    # channel attention module
    def __init__(self, channel, stride=1, reduction=8, bias=True):
        super(ChannelAtt, self).__init__()
        self.channel = channel
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale
        self.channel_down = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // reduction, (1, 1), padding=0, bias=bias),
            nn.PReLU()
        )
        # feature upscale --> channel weight
        self.channel_up1 = nn.Sequential(
            nn.Conv2d(self.channel // reduction, self.channel, (1, 1), padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.channel_up2 = nn.Sequential(
            nn.Conv2d(self.channel // reduction, self.channel, (1, 1), padding=0, bias=bias),
            nn.Sigmoid()
        )
        # different resolution to same
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, (3, 3), stride=stride,
                               padding=(1, 1), output_padding=(1, 1), bias=bias),
            nn.PReLU()
        )

    def forward(self, x, y):
        fusion = torch.add(x, y)
        fusion = self.channel_down(self.avg_pool(fusion))
        out_x = self.channel_up1(fusion)
        out_y = self.channel_up2(fusion)
        return [out_x, out_y]


class SpatialAtt(nn.Module):
    # spatial attention module
    def __init__(self, channel, stride=1, kernel=(3, 3), padding=(1, 1), bias=True):
        super(SpatialAtt, self).__init__()
        self.channel = channel
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=bias),
            # nn.BatchNorm2d(in_channel, eps=1e-5, momentum=0.01, affine=True),
            nn.PReLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2), padding=padding, bias=bias),
            nn.PReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2),
                               padding=padding, output_padding=(1, 1), bias=bias),
            nn.Sigmoid()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2),
                               padding=padding, output_padding=(1, 1), bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        fusion = torch.cat([x, y], dim=1)
        fusion = self.down(self.conv_fusion(fusion))
        up_x = self.up1(fusion)
        up_y = self.up2(fusion)
        return [up_x, up_y]


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3):
        super(FusionBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.SpaBranch = DenseLayer(self.in_channels, self.out_channels)
        self.FreBranch = FreLayer(self.in_channels, self.out_channels)
        self.spa_post = nn.Conv2d((self.in_channels + self.out_channels) * 2, self.out_channels, (3, 3), (1, 1), 1, bias=True)

        self.spa_att = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels // 2, (3, 3), (1, 1), 1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(self.out_channels // 2, self.out_channels, (3, 3), (1, 1), 1, bias=True),
                                     nn.Sigmoid())

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(self.out_channels * 2, self.out_channels // 2, (1, 1), (1, 1), 0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(self.out_channels // 2, self.out_channels * 2, (1, 1), (1, 1), 0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(self.out_channels * 2, self.out_channels, (3, 3), (1, 1), 1, bias=True)

        self.channel_map = ChannelAtt(self.out_channels)
        self.spatial_map = SpatialAtt(self.out_channels)

    def forward(self, over, under):
        _, _, H, W = over.shape

        spa_over = self.SpaBranch(over)
        spa_under = self.SpaBranch(under)
        spa_output = self.spa_post(torch.cat([spa_over, spa_under], dim=1))
        fre_output = self.FreBranch(over, under)

        # spa_map = self.spa_att(spa_output - fre_output)
        # spa_res = fre_output * spa_map + spa_output
        # cat_f = torch.cat([spa_res, fre_output], 1)
        # fusion = self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f)

        fusion_x = spa_output * self.channel_map(spa_output, fre_output)[0] * self.spatial_map(spa_output, fre_output)[0]
        fusion_y = fre_output * self.channel_map(spa_output, fre_output)[1] * self.spatial_map(spa_output, fre_output)[1]
        fusion = fusion_x + fusion_y

        return fusion, spa_over, spa_under, fre_output


class Fusion(nn.Module):
    def __init__(self, channels, growth=8):
        super(Fusion, self).__init__()
        self.growth = growth
        self.channels = channels

        self.over_input = nn.Sequential(nn.Conv2d(1, self.channels, (3, 3), (1, 1), 1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                        nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1), 0)
                                        )
        self.under_input = nn.Sequential(nn.Conv2d(1, self.channels, (3, 3), (1, 1), 1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.LeakyReLU(0.1),
                                         nn.Conv2d(self.channels, self.channels, (3, 3), (1, 1), 1),
                                         nn.Conv2d(self.channels, self.channels, (1, 1), (1, 1), 0)
                                         )
        # modules = []
        # for i in range(5):
        #     modules.append(FusionBlock(self.channels, self.growth))
        #     self.channels += self.growth
        # self.fusion_layer = nn.Sequential(*modules)

        self.fusion_layer1 = FusionBlock(self.channels, self.growth)
        self.fusion_layer2 = FusionBlock(self.channels + self.growth, self.growth)
        self.fusion_layer3 = FusionBlock(self.channels + self.growth * 2, self.growth)
        self.fusion_layer4 = FusionBlock(self.channels + self.growth * 3, self.growth)
        # self.fusion_layer5 = FusionBlock(self.channels + self.growth * 4, self.growth)

        self.output = nn.Sequential(
            ConvBlock(self.growth * 4, 128, kernel_size=3, act_type='lrelu', norm_type=None),
            ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
            ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
            ConvBlock(32, 8, kernel_size=3, act_type='lrelu', norm_type=None),
            nn.Conv2d(8, 1, (3, 3), (1, 1), 1),
            nn.Tanh()
        )

        self.fre_loss = nn.Conv2d(self.growth, 1, (1, 1), (1, 1), 0, bias=True)

    def forward(self, over, under):
        over = self.over_input(over)
        under = self.under_input(under)

        fusion1, spa_over1, spa_under1, fre1 = self.fusion_layer1(over, under)
        fusion2, spa_over2, spa_under2, fre2 = self.fusion_layer2(spa_over1, spa_under1)
        fusion3, spa_over3, spa_under3, fre3 = self.fusion_layer3(spa_over2, spa_under2)
        fusion4, spa_over4, spa_under4, fre4 = self.fusion_layer4(spa_over3, spa_under3)
        # fusion5, spa_over5, spa_under5 = self.fusion_layer5(spa_over4, spa_under4)

        fusion = torch.cat([fusion1, fusion2, fusion3, fusion4], dim=1)
        fusion = self.output(fusion)

        fre_output1 = self.fre_loss(fre1)
        fre_output2 = self.fre_loss(fre2)
        fre_output3 = self.fre_loss(fre3)
        fre_output4 = self.fre_loss(fre4)

        return fusion, fre_output1, fre_output2, fre_output3, fre_output4


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = nn.Sequential(*[self.features[i] for i in range(3)])(x)
        x_1 = self.features[3](x)
        x_2 = nn.Sequential(*[self.features[i] for i in range(4, 8)])(x_1)
        x_2 = self.features[8](x_2)
        x_3 = nn.Sequential(*[self.features[i] for i in range(8, 15)])(x_2)
        x_3 = self.features[15](x_3)
        x_4 = nn.Sequential(*[self.features[i] for i in range(15, 22)])(x_3)
        x_4 = self.features[22](x_4)
        x_5 = nn.Sequential(*[self.features[i] for i in range(22, 29)])(x_4)
        x_5 = self.features[29](x_5)
        return x_1, x_2, x_3, x_4, x_5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Sigmoid()]
            else:
                layers += [conv2d, nn.Sigmoid()]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
        model.load_state_dict(torch.load('vgg16.pth'))
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
