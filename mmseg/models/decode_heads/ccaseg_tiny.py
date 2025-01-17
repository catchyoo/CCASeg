import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from ..utils import resize
from ..builder import MODELS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import math
from mmengine.model import BaseModule
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model.weight_init import trunc_normal_init
from mmcv.cnn import build_activation_layer, build_norm_layer

#----------------------------------------------------------------------------------------------------------------#

class InjectionMultiSum(nn.Module):
    def __init__(
        self, dim, norm_cfg=dict(type='BN', requires_grad=True), activations = None,) -> None:
        super(InjectionMultiSum, self).__init__()

        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(dim, dim, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_act = ConvModule(dim, dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(global_act, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + sig_act
        return out

class Inject(nn.Module):
    def __init__(
        self, dim1, dim2, norm_cfg=dict(type='BN', requires_grad=True), activations = None,) -> None:
        super(Inject, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(dim1, dim2, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(dim2, dim2, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(global_act, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + sig_act
        return out

#----------------------------------------------------------------------------------------------------------------#

class LKCA2(BaseModule):
    def __init__(self, channels1, channels2, norm_cfg=dict(type='SyncBN', requires_grad=True), act_cfg=dict(type='GELU'), 
                 kernel_sizes=[3, [1, 5], [1, 9], [1, 13]], paddings=[1, [0, 2], [0, 4], [0, 6]]):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, channels1)[1]
        self.norm2 = build_norm_layer(norm_cfg, channels2)[1]

        self.proj1 = nn.Conv2d(channels1, channels1, kernel_size=1)
        self.proj2 = nn.Conv2d(channels2, channels1, kernel_size=1)

        self.activation1 = build_activation_layer(act_cfg)
        self.activation2 = build_activation_layer(act_cfg)
        
        self.conv1 = nn.Conv2d(channels1, channels1, kernel_size=kernel_sizes[0], padding=paddings[0], groups=channels1)
        self.conv2 = nn.Conv2d(channels1, channels1, kernel_size=kernel_sizes[0], padding=paddings[0], groups=channels1)

        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']  # Local
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,conv_name):
                self.add_module(i_conv, nn.Conv2d(channels1, channels1, tuple(i_kernel), padding=i_pad, groups=channels1))

        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_3', f'conv{i}_4']  # Global
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,conv_name):
                self.add_module(i_conv, nn.Conv2d(channels1, channels1, tuple(i_kernel), padding=i_pad, groups=channels1))

        self.conv3 = nn.Conv2d(channels1, channels1, 1)

        # MLP
        self.norm3 = build_norm_layer(norm_cfg, channels1)[1]
        self.conv4 = nn.Conv2d(channels1, channels1, 1)
        self.dw = nn.Conv2d(channels1, channels1, kernel_size=3, stride=1, padding=1, bias=True, groups=channels1)
        self.activation3 = build_activation_layer(act_cfg)
        self.conv5 = nn.Conv2d(channels1, channels1, 1)

    def forward(self, x1, x2):
        shorcut = x1.clone()

        x1 = self.norm1(x1)
        x1 = self.proj1(x1)
        x1 = self.activation1(x1)

        x2 = self.norm2(x2)
        x2 = self.proj2(x2)
        x2 = self.activation2(x2)

        attn1 = self.conv1(x1)
        attn2 = self.conv2(x2)

        # Multi-Scale Feature extraction : x2
        attn1_0 = self.conv0_1(attn1)
        attn1_0 = self.conv0_2(attn1_0)

        attn1_1 = self.conv1_1(attn1)
        attn1_1 = self.conv1_2(attn1_1)

        attn1_2 = self.conv2_1(attn1)
        attn1_2 = self.conv2_2(attn1_2)

        # Multi-Scale Feature extraction : x2
        attn2_0 = self.conv0_3(attn2)
        attn2_0 = self.conv0_4(attn2_0)

        attn2_1 = self.conv1_3(attn2)
        attn2_1 = self.conv1_4(attn2_1)

        attn2_2 = self.conv2_3(attn2)
        attn2_2 = self.conv2_4(attn2_2)

        attn = attn1*attn2 + attn1_0*attn2_0 + attn1_1*attn2_1 + attn1_2*attn2_2

        # Channel Mixing
        attn = self.conv3(attn)
        x = x1 * attn
        x = self.conv3(x) + shorcut

        out = self.norm3(x)
        out = self.conv4(out)
        out = self.dw(out)
        out = self.activation3(out)
        out = self.conv5(out)

        res = out + x
        return res
    
class LKCA3(BaseModule):
    def __init__(self, channels1, channels2, norm_cfg=dict(type='SyncBN', requires_grad=True), act_cfg=dict(type='GELU'), 
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]], paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, channels1)[1]
        self.norm2 = build_norm_layer(norm_cfg, channels2)[1]

        self.proj1 = nn.Conv2d(channels1, channels1, kernel_size=1)
        self.proj2 = nn.Conv2d(channels2, channels1, kernel_size=1)

        self.activation1 = build_activation_layer(act_cfg)
        self.activation2 = build_activation_layer(act_cfg)
        
        self.conv1 = nn.Conv2d(channels1, channels1, kernel_size=kernel_sizes[0], padding=paddings[0], groups=channels1)
        self.conv2 = nn.Conv2d(channels1, channels1, kernel_size=kernel_sizes[0], padding=paddings[0], groups=channels1)

        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']  # Local
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,conv_name):
                self.add_module(i_conv, nn.Conv2d(channels1, channels1, tuple(i_kernel), padding=i_pad, groups=channels1))

        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_3', f'conv{i}_4']  # Global
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,conv_name):
                self.add_module(i_conv, nn.Conv2d(channels1, channels1, tuple(i_kernel), padding=i_pad, groups=channels1))

        self.conv3 = nn.Conv2d(channels1, channels1, 1)

        # MLP
        self.norm3 = build_norm_layer(norm_cfg, channels1)[1]
        self.conv4 = nn.Conv2d(channels1, channels1, 1)
        self.dw = nn.Conv2d(channels1, channels1, kernel_size=3, stride=1, padding=1, bias=True, groups=channels1)
        self.activation3 = build_activation_layer(act_cfg)
        self.conv5 = nn.Conv2d(channels1, channels1, 1)

    def forward(self, x1, x2):
        shorcut = x1.clone()

        x1 = self.norm1(x1)
        x1 = self.proj1(x1)
        x1 = self.activation1(x1)

        x2 = self.norm2(x2)
        x2 = self.proj2(x2)
        x2 = self.activation2(x2)

        attn1 = self.conv1(x1)
        attn2 = self.conv2(x2)

        # Multi-Scale Feature extraction : x2
        attn1_0 = self.conv0_1(attn1)
        attn1_0 = self.conv0_2(attn1_0)

        attn1_1 = self.conv1_1(attn1)
        attn1_1 = self.conv1_2(attn1_1)

        attn1_2 = self.conv2_1(attn1)
        attn1_2 = self.conv2_2(attn1_2)

        # Multi-Scale Feature extraction : x2
        attn2_0 = self.conv0_3(attn2)
        attn2_0 = self.conv0_4(attn2_0)

        attn2_1 = self.conv1_3(attn2)
        attn2_1 = self.conv1_4(attn2_1)

        attn2_2 = self.conv2_3(attn2)
        attn2_2 = self.conv2_4(attn2_2)

        attn = attn1*attn2 + attn1_0*attn2_0 + attn1_1*attn2_1 + attn1_2*attn2_2

        # Channel Mixing
        attn = self.conv3(attn)
        x = x1 * attn
        x = self.conv3(x) + shorcut

        out = self.norm3(x)
        out = self.conv4(out)
        out = self.dw(out)
        out = self.activation3(out)
        out = self.conv5(out)

        res = out + x
        return res
    
#----------------------------------------------------------------------------------------------------------------#
       
class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, dim):
        super().__init__()
        self.stride = stride
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        #self.norm = nn.LayerNorm(dim)
        #self.act = nn.GELU()

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        _H = H // self.stride
        _W = W // self.stride
        out = self.sr(nn.functional.adaptive_avg_pool2d(inputs, (_H, _W)))
        #w_ = self.norm(w_)
        #out = self.act(w_)
        return out

#----------------------------------------------------------------------------------------------------------------#

@MODELS.register_module()
class CCASegHead_tiny(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(CCASegHead_tiny, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        # F2-Pooling Module
        self.down2_1 = PyramidPoolAgg(stride=2, dim=c2_in_channels) # feature size = 32*32
        self.down2_2 = PyramidPoolAgg(stride=4, dim=c2_in_channels) # feature size = 16*16

        # F3-Pooling Module
        self.down3_1 = PyramidPoolAgg(stride=2, dim=c3_in_channels) # feature size = 16*16


        self.attnc4 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels) # feature size = 16*16
        self.attnc3 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels) # feature size = 32*32
        self.attnc2 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels) # feature size = 64*64

        self.attnc4_1 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels) # feature size = 16*16
        self.attnc3_1 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels) # feature size = 32*32
        self.attnc2_1 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels) # feature size = 64*64

        self.attnc4_2 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels) # feature size = 16*16
        self.attnc3_2 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels) # feature size = 32*32
        self.attnc2_2 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels) # feature size = 64*64

        self.attnc4_3 = LKCA2(channels1=c4_in_channels, channels2=c4_in_channels) # feature size = 16*16
        self.attnc3_3 = LKCA3(channels1=c3_in_channels, channels2=c4_in_channels) # feature size = 32*32
        self.attnc2_3 = LKCA3(channels1=c2_in_channels, channels2=c4_in_channels) # feature size = 64*64

        # Injection Block(Residual = high feature)
        self.inject2 = InjectionMultiSum(dim=c2_in_channels)
        self.inject3 = InjectionMultiSum(dim=c3_in_channels)
        self.inject4 = InjectionMultiSum(dim=c4_in_channels)

        # Injection Block(Residual = high feature)
        self.multi8_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi8_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)

        self.multi16_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi16_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)

        self.multi32_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi32_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)

        self.multi64_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.multi64_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)

        self.predict2_3 = Inject(dim1=c2_in_channels, dim2=c3_in_channels)
        self.predict3_4 = Inject(dim1=c3_in_channels, dim2=c4_in_channels)

        self.linear_fuse = ConvModule(
            in_channels=(c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape # 16x16
        _, _, h3, w3 = c3.shape # 32x32
        _, _, h2, w2 = c2.shape # 64x64

        # F2-Pooling
        c2_1 = self.down2_1(c2) # 32x32
        _, _, c2_1_h1, c2_1_w1 = c2_1.shape # 32x32
        c2_2 = self.down2_2(c2) # 16x16

        # F3-Pooling
        c3_1 = self.down3_1(c3) # 16x16

        # F4 rectification 16x16 => Injection
        key_4 = self.multi8_3(c2_2, c3_1)
        key_4 = self.multi8_4(key_4, c4)

        _c4 = self.attnc4(c4, key_4)
        _c4 = self.attnc4_1(_c4, key_4)
        _c4 = self.attnc4_2(_c4, key_4)
        _c4 = self.attnc4_3(_c4, key_4)

        new4 = self.inject4(c4, _c4) # size of new4 : 16x16

        # F3 rectification 32x32 => Injection
        new4_up32 = resize(new4, size=(c2_1_h1, c2_1_w1), mode='bilinear', align_corners=False)
        key_3 = self.multi16_3(c2_1, c3)
        key_3 = self.multi16_4(key_3, new4_up32)

        _c3 = self.attnc3(c3, key_3)
        _c3 = self.attnc3_1(_c3, key_3)
        _c3 = self.attnc3_2(_c3, key_3)
        _c3 = self.attnc3_3(_c3, key_3)

        new3 = self.inject3(c3, _c3) # size of new3 : 32x32

        # F2 rectification 64x64 => Injection
        new3_up64 = resize(new3, size=(h2, w2), mode='bilinear', align_corners=False)
        new4_up64 = resize(new4, size=(h2, w2), mode='bilinear', align_corners=False)
        key_2 = self.multi32_3(c2, new3_up64)
        key_2 = self.multi32_4(key_2, new4_up64)

        _c2 = self.attnc2(c2, key_2)
        _c2 = self.attnc2_1(_c2, key_2)
        _c2 = self.attnc2_2(_c2, key_2)        
        _c2 = self.attnc2_3(_c2, key_2)

        new2 = self.inject2(c2, _c2)

        pred2 = self.predict2_3(new2,new3)
        pred3 = self.predict3_4(pred2,new4)

        _c = self.linear_fuse(pred3)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x