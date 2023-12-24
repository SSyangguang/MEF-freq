import warnings
import torch
import torch.nn as nn
import math
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

        fusion_x = spa_output * self.channel_map(spa_output, fre_output)[0] * self.spatial_map(spa_output, fre_output)[0]
        fusion_y = fre_output * self.channel_map(spa_output, fre_output)[1] * self.spatial_map(spa_output, fre_output)[1]
        fusion = fusion_x + fusion_y

        return fusion, spa_over, spa_under, fre_output, spa_output


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

        self.fusion_layer1 = FusionBlock(self.channels, self.growth)
        self.fusion_layer2 = FusionBlock(self.channels + self.growth, self.growth)
        self.fusion_layer3 = FusionBlock(self.channels + self.growth * 2, self.growth)
        self.fusion_layer4 = FusionBlock(self.channels + self.growth * 3, self.growth)

        self.output = nn.Sequential(
            ConvBlock(self.growth * 4, 128, kernel_size=3, act_type='lrelu', norm_type=None),
            ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
            ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
            ConvBlock(32, 8, kernel_size=3, act_type='lrelu', norm_type=None),
            nn.Conv2d(8, 1, (3, 3), (1, 1), 1),
            nn.Tanh()
        )

    def forward(self, over, under):
        over = self.over_input(over)
        under = self.under_input(under)

        fusion1, spa_over1, spa_under1, fre1, spa1 = self.fusion_layer1(over, under)
        fusion2, spa_over2, spa_under2, fre2, spa2 = self.fusion_layer2(spa_over1, spa_under1)
        fusion3, spa_over3, spa_under3, fre3, spa3 = self.fusion_layer3(spa_over2, spa_under2)
        fusion4, spa_over4, spa_under4, fre4, spa4 = self.fusion_layer4(spa_over3, spa_under3)

        fusion = torch.cat([fusion1, fusion2, fusion3, fusion4], dim=1)
        fusion = self.output(fusion)

        return fusion