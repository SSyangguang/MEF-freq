import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from option import args


'''FCA Net block'''
def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        # 如果输入大小和DCT大小不一样的话，我们会先做个resize然后在提取DCT频谱，而且resize只对应提取频谱，不会对输入特征造成任何影响。
        # 其实也有个简单的办法，你可以把所有的dct_w和dct_h统统都设置为7，不用考虑输入大小的问题。
        if h != self.dct_h or w != self.dct_w:
            # 让x的尺寸经过自适应平均池化后，尺寸变为(self.dct_h, self.dct_w)
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        # expand_as将y扩展为和x一样的维度
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        # register_buffer定义的参数不能更新
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        # 切分的通道数量
        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 对通道进行切分
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


'''Res FFT block'''
class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1))
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1)),
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y



'''baseline'''


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=(3, 3), bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.feat_num = args.feature_num
        self.padding = kernel[0] // 2
        self.bn = bn

        feat_layers = list()
        feat_layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=(1, 1), padding=self.padding, bias=bias))
        if self.bn:
            feat_layers.append(nn.BatchNorm2d(out_channel))
        feat_layers.append(nn.PReLU())

        self.feat_extr = nn.Sequential(*feat_layers)

    def forward(self, x):
        return self.feat_extr(x)


class Input(nn.Module):
    def __init__(self, in_channel=1, kernel=(3, 3), bn=False, bias=True):
        super(Input, self).__init__()
        self.channel = args.feature_num
        self.bn = bn
        padding = kernel[0] // 2

        layers = list()
        layers.append(nn.Conv2d(in_channel, self.channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias))
        if self.bn:
            layers.append(nn.BatchNorm2d(self.channel))
        layers.append(nn.PReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.main(x)
        return feat


class Output(nn.Module):
    def __init__(self, in_channel, out_channel=1, kernel=(3, 3), bias=True):
        super(Output, self).__init__()
        self.channel = in_channel
        padding = kernel[0] // 2

        layers = list()
        layers.append(nn.Conv2d(self.channel, self.channel // 2, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias))
        layers.append(nn.Conv2d(self.channel // 2, out_channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        output = self.main(x)
        return output


class ChannelAtt(nn.Module):
    # channel attention module
    def __init__(self, channel, reduction=8, bias=True):
        super(ChannelAtt, self).__init__()
        self.channel = channel
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale
        self.channel_down = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // reduction, (1, 1), padding=(0, 0), bias=bias),
            nn.PReLU()
        )
        # feature upscale --> channel weight
        self.channel_up1 = nn.Sequential(
            nn.Conv2d(self.channel // reduction, self.channel, (1, 1), padding=(0, 0), bias=bias),
            nn.Sigmoid()
        )
        self.channel_up2 = nn.Sequential(
            nn.Conv2d(self.channel // reduction, self.channel, (1, 1), padding=(0, 0), bias=bias),
            nn.Sigmoid()
        )

        # # 不同的通道数量对应一种dct的大小
        # c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        # # 这里的plane应该就是第一个卷积层输出的通道数量
        # self.planes = planes
        # self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes], reduction=reduction,
        #                                        freq_sel_method='top16')

    def forward(self, x, y):
        fusion = torch.add(x, y)
        fusion = self.channel_down(self.avg_pool(fusion))
        out_x = self.channel_up1(fusion)
        out_y = self.channel_up2(fusion)
        return [out_x, out_y]


class SpatialAtt(nn.Module):
    # spatial attention module
    def __init__(self, channel, padding=(1, 1), bias=True):
        super(SpatialAtt, self).__init__()
        self.channel = channel
        self.down = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2), padding=padding, bias=bias),
            nn.PReLU()
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias),
            # nn.BatchNorm2d(in_channel, eps=1e-5, momentum=0.01, affine=True),
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


class FusionBlock(nn.Module):
    def __init__(self, channel):
        super(FusionBlock, self).__init__()
        self.channel = args.feature_num
        self.channel_map = ChannelAtt(channel=channel)
        self.spatial_map = SpatialAtt(channel=channel)

    def forward(self, x, y):
        fusion_x = x * self.channel_map(x, y)[0] * self.spatial_map(x, y)[0]
        fusion_y = y * self.channel_map(x, y)[1] * self.spatial_map(x, y)[1]
        fusion = fusion_x + fusion_y
        return fusion


class baseline(nn.Module):
    def __init__(self, depth=3, in_channel=1, out_channel=1, kernel=(3, 3), bn=False):
        super(baseline, self).__init__()
        self.depth = depth
        self.channel = args.feature_num
        self.padding = kernel[0] // 2
        self.bn = bn

        self.input = Input(in_channel=in_channel, bn=True)
        self.feat_extr = nn.ModuleList([nn.Sequential(BasicConv(self.channel * (i+1), self.channel))
                                        for i in range(self.depth)])
        self.fusion = FusionBlock(self.channel*(depth+1))
        self.output = Output(self.channel*(depth+1), out_channel)

    def forward(self, ir, vis):
        ir_input = self.input(ir)
        vis_input = self.input(vis)

        for i in range(self.depth):
            ir_dense = self.feat_extr[i](ir_input)
            ir_input = torch.cat([ir_input, ir_dense], dim=1)
            vis_dense = self.feat_extr[i](vis_input)
            vis_input = torch.cat([vis_input, vis_dense], dim=1)

        fusion_fea = self.fusion(ir_input, vis_input)
        output = self.output(fusion_fea)
        output = output / 2 + 0.5

        return output, ir_input, vis_input, fusion_fea

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class FFTonly(nn.Module):
    def __init__(self, depth=3, in_channel=1, out_channel=1, kernel=(3, 3), bn=False):
        super(FFTonly, self).__init__()
        self.depth = depth
        self.channel = args.feature_num
        self.padding = kernel[0] // 2
        self.bn = bn

        self.input = Input(in_channel=in_channel, bn=True)
        self.feat_extr = nn.ModuleList([nn.Sequential(BasicConv(self.channel * (i+1), self.channel))
                                        for i in range(self.depth)])
        self.fft = ResBlock_fft_bench(self.channel)
        self.fusion = FusionBlock(self.channel*(depth+1))
        self.output = Output(self.channel*(depth+1), out_channel)

    def forward(self, ir, vis):
        ir_input = self.input(ir)
        vis_input = self.input(vis)

        for i in range(self.depth):
            ir_dense = self.feat_extr[i](ir_input)
            ir_dense = self.fft(ir_dense)
            ir_input = torch.cat([ir_input, ir_dense], dim=1)
            vis_dense = self.feat_extr[i](vis_input)
            vis_dense = self.fft(vis_dense)
            vis_input = torch.cat([vis_input, vis_dense], dim=1)

        fusion_fea = self.fusion(ir_input, vis_input)
        output = self.output(fusion_fea)
        output = output / 2 + 0.5

        return output, ir_input, vis_input, fusion_fea

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

