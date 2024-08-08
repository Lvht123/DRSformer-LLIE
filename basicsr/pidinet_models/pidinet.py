"""
Author: Zhuo Su, Wenzhe Liu
Date: Feb 18, 2021
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import Conv2d
from .config import config_model, config_model_converted
from einops import rearrange
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .SS2D_arch import SS2D
from .IFA_arch import IFA
from basicsr.models.transformer import BasicUformerLayer, Downsample, InputProj, OutputProj
from basicsr.models.SAG import ConvLayer,ConvLayer2
class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y
    
class PreNorm(nn.Module):
    """
    预归一化模块，通常用于Transformer架构中。

    在执行具体的功能（如自注意力或前馈网络）之前先进行层归一化，
    这有助于稳定训练过程并提高模型性能。

    属性:
        dim: 输入特征的维度。
        fn: 要在归一化后应用的模块或函数。
    """

    def __init__(self, dim, fn):
        """
        初始化预归一化模块。

        参数:
            dim (int): 输入特征的维度，也是层归一化的维度。
            fn (callable): 在归一化之后应用的模块或函数。
        """
        super().__init__()  # 初始化基类 nn.Module
        self.fn = fn  # 存储要应用的函数或模块
        self.norm = nn.LayerNorm(dim)  # 创建层归一化模块

    def forward(self, x, *args, **kwargs):
        """
        对输入数据进行前向传播。

        参数:
            x (Tensor): 输入到模块的数据。
            *args, **kwargs: 传递给self.fn的额外参数。

        返回:
            Tensor: self.fn的输出，其输入是归一化后的x。
        """
        x = self.norm(x)  # 首先对输入x进行层归一化
        return self.fn(x, *args, **kwargs)  # 将归一化的数据传递给self.fn，并执行
class FeedForward(nn.Module):
    """
    实现一个基于卷积的前馈网络模块，通常用于视觉Transformer结构中。
    这个模块使用1x1卷积扩展特征维度，然后通过3x3卷积在这个扩展的维度上进行处理，最后使用1x1卷积将特征维度降回原来的大小。

    参数:
        dim (int): 输入和输出特征的维度。
        mult (int): 特征维度扩展的倍数，默认为4。
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  # 使用1x1卷积提升特征维度
            GELU(),  # 使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),  # 分组卷积处理，维持特征维度不变，增加特征的局部相关性
            GELU(),  # 再次使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),  # 使用1x1卷积降低特征维度回到原始大小
        )

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        x (tensor): 输入特征，形状为 [b, h, w, c]，其中b是批次大小，h和w是空间维度，c是通道数。

        返回:
        out (tensor): 输出特征，形状与输入相同。
        """
        # 由于PyTorch的卷积期望的输入形状为[b, c, h, w]，需要将通道数从最后一个维度移到第二个维度
        out = self.net(x.permute(0, 3, 1, 2).contiguous())  # 调整输入张量的维度
        return out.permute(0, 2, 3, 1)  # 将输出张量的维度调整回[b, h, w, c]格式    
class GELU(nn.Module):
    """
    GELU激活函数的封装。

    GELU (Gaussian Error Linear Unit) 是一种非线性激活函数，
    它被广泛用于自然语言处理和深度学习中的其他领域。
    这个函数结合了ReLU和正态分布的性质。
    """

    def forward(self, x):
        """
        在输入数据上应用GELU激活函数。

        参数:
            x (Tensor): 输入到激活函数的数据。

        返回:
            Tensor: 经过GELU激活函数处理后的数据。
        """
        return F.gelu(x)  # 使用PyTorch的函数实现GELU激活
    
class CDCM1(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM1, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4
class CDCM2(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM2, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4
# class CDCM(nn.Module):
#     """
#     Compact Dilation Convolution based Module
#     """
#     def __init__(self,in_channel, out_channel, img_size=128,
#                  kernel_size=3, depth=2, num_head=2, win_size=8, mlp_ratio=4,
#                  qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, patch_norm=True,
#                  use_checkpoint=False, token_projection='linear', token_mlp='leff', shift_flag=True,
#                  downsample=False, device='cpu'):
#         super(CDCM, self).__init__()
#         self.conv0 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample, device=device)
#         # self.relu1 = nn.ReLU()
#         # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
#         # self.conv2_1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=5, padding=5, bias=False)
#         # self.conv2_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=7, padding=7, bias=False)
#         # self.conv2_3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=9, padding=9, bias=False)
#         # self.conv2_4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=11, padding=11, bias=False)
#         # nn.init.constant_(self.conv1.bias, 0)
#         self.transformer0 = BasicUformerLayer(dim=in_channel,
#                                               output_dim=in_channel,
#                                               input_resolution=(img_size,
#                                                                 img_size),
#                                               depth=depth,
#                                               num_heads=num_head,
#                                               win_size=win_size,
#                                               mlp_ratio=mlp_ratio,
#                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                               drop=drop_rate, attn_drop=attn_drop_rate,
#                                               norm_layer=norm_layer,
#                                               use_checkpoint=use_checkpoint,
#                                               token_projection=token_projection, token_mlp=token_mlp,
#                                               shift_flag=shift_flag)

#         ####
#         self.input_proj0 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
#         self.output_proj0 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
#         # self.input_proj1 = InputProj(in_channel=in_channel, out_channel=in_channel, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
#         # self.output_proj1 = OutputProj(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1)
#         # self.fusion0 = ConvLayer(out_channel * 3, out_channel, 1, device=device)
#         # self.downsample0 = Downsample(in_channel, in_channel)
        
#     def forward(self, input):
#         # x = self.relu1(input)
#         # x = self.conv1(x)
#         short_range = self.conv0(input)
#         # x2 = self.conv2_2(x)
#         # x3 = self.conv2_3(x)
#         # x4 = self.conv2_4(x)
#         # short_range = self.conv0(input)
#         # print(input.shape)
#         input_long = self.input_proj0(input)
#         # print(input_long1.shape)
#         long_range = self.transformer0(input_long)
#         # long_range1 = self.conv2_3(input_long1)
#         long_range = self.output_proj0(long_range)
#         # input_long2 = self.input_proj1(input)
#         # long_range = self.transformer0(input_long2)
#         # long_range2 = self.conv2_4(input_long2)
#         # long_range2 = self.output_proj1(long_range2)
        
#         # f0 = self.fusion0(torch.cat([short_range1, long_range1,long_range2], dim=1))
#         return short_range + long_range

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride=stride
            
        self.stride=stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride=stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, self.inplane, 
                    kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 2C
        
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane) # 4C

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            # for i in range(4):
            #     self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
            #     self.attentions.append(CSAM(self.dil))
            #     self.conv_reduces.append(MapReduce(self.dil))
            self.dilations.append(CDCM1(self.fuseplanes[0], self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))
            self.dilations.append(CDCM1(self.fuseplanes[1], self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))
            self.dilations.append(CDCM2(self.fuseplanes[2], self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))
            self.dilations.append(CDCM2(self.fuseplanes[3], self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                # self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                # self.conv_reduces.append(MapReduce(self.dil))
                self.dilations.append(CDCM1(self.fuseplanes[0], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
                self.dilations.append(CDCM1(self.fuseplanes[1], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
                self.dilations.append(CDCM2(self.fuseplanes[2], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
                self.dilations.append(CDCM2(self.fuseplanes[3], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1) # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        output = torch.sigmoid(output)
        #if not self.training:
        #    return torch.sigmoid(output)

        # outputs.append(output)
        # outputs = [torch.sigmoid(r) for r in outputs]
        return output


def pidinet_tiny(args):
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa)

def pidinet_small(args):
    pdcs = config_model(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa)

def pidinet(args):
    pdcs = config_model(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa)



## convert pidinet to vanilla cnn

def pidinet_tiny_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_small_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa, convert=True)
