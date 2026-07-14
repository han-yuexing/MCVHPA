"""
Author: Zhuo Su, Wenzhe Liu
Date: Feb 18, 2021
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .ops import Conv2d
from models.config import config_model, config_model_converted
import matplotlib.pyplot as plt 
import torchvision
from .PartialConv2d import PartialConv2d


def featuremap_visual(feature,
                      feture_name,
                      out_dir="//root//segment3//my_model//feature_map//",  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=False,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      num_ch= -1,  # 显示特征图前几个通道，-1 or None 都显示
                      nrow=20,  # 每行显示多少个特征图通道
                      padding=30,  # 特征图之间间隔多少像素值
                      pad_value=30,  # 特征图之间的间隔像素
                      stage = 1
                      ):
    # feature = feature.detach().cpu()
    root,img_name = os.path.split(feture_name[0])

    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)

    if c > num_ch > 0:
        feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    img_max = img.max()
    img = (img/img_max) * 255
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) + '-' + str(img_name)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title)
        plt.savefig(out_root,dpi = 1000)



    if show_feature:        plt.show()
    
# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x):
#         x = self.dwconv(x)
#         return x



# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
#         self.drop = nn.Dropout(drop)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.dwconv(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


    
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # print(x.shape)
        # print('Attention')
        shorcut = x.clone()
        # print(shorcut.shape)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = x + shorcut
        x = self.proj_2(x)
        # print(x.shape)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn




class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.GELU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        # print("CSAM")
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        # print(y.shape)
        # print((x*y).shape)

        return x * y

class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        # self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        # self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        # self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        # self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        self.conv2_1 = PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, multi_channel=True)
        self.conv2_2 = PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, multi_channel=True)
        self.conv2_3 = PartialConv2d(out_channels, out_channels, kernel_size=5, padding=2, bias=False, multi_channel=True)
        self.conv2_4 = PartialConv2d(out_channels, out_channels, kernel_size=7, padding=3, bias=False, multi_channel=True)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        # print(x.shape)
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        # x1 = self.bn1(x1)
        # x1 = self.relu1(x1)
        x2 = self.conv2_2(x)
        # x1 = self.bn1(x2)
        # x1 = self.relu1(x2)
        x3 = self.conv2_3(x)
        # x1 = self.bn1(x3)
        # x1 = self.relu1(x3)
        x4 = self.conv2_4(x)
        # x1 = self.bn1(x4)
        # x1 = self.relu1(x4)
        # print(x4.shape)
        # print((x1 + x2 + x3 + x4).shape)
        return x1 + x2 + x3 + x4


class CDCM_ASPP(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels, out_channels):
        super(CDCM_ASPP, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(240, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        # print(x.shape)
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x1 = self.bn1(x1)
        # print(x1.shape)
        x2 = self.conv2_2(x)
        x2 = self.bn1(x2)
        # print(x2.shape)
        x3 = self.conv2_3(x)
        x3 = self.bn1(x3)
        # print(x3.shape)
        x4 = self.conv2_4(x)
        x4 = self.bn1(x4)
        # print(x4.shape)
        # print(x4.shape)
        # print((x1 + x2 + x3 + x4).shape)
        return x1 + x2 + x3 + x4


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
            self.bn = nn.BatchNorm2d(inplane)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.GELU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # print("PDCBlock")
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
        # print("PDCBlock_converted")
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PiDiNet_new(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False, img_name = None):
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
            for i in range(3):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(Attention(self.dil))
                # self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(3):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(3):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(3):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(3, 1, kernel_size=1) # has bias
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
    

    def forward(self, x, x_name):
        # print('测试4——1')
        x_feature = x
        
        H, W = x.size()[2:]
        
        # print('H, W:',H, W)

        x = self.init_block(x)
        
        


        x1 = self.block1_1(x)
        
        

        x1 = self.block1_2(x1)
        

        x1 = self.block1_3(x1)
        
        
        x1_feature = x1
        # featuremap_visual(x1_feature)


        x2 = self.block2_1(x1)
        # featuremap_visual(x2)
        x2 = self.block2_2(x2)
        # featuremap_visual(x2)
        x2 = self.block2_3(x2)
        # featuremap_visual(x2)
        x2 = self.block2_4(x2)
        
        x2_feature = x2
        # featuremap_visual(x2_feature)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)
        
        x3_feature = x3
        # featuremap_visual(x3_feature)

        # x4 = self.block4_1(x3)
        # x4 = self.block4_2(x4)
        # x4 = self.block4_3(x4)
        # x4 = self.block4_4(x4)

 
        # x4_feature = x4
        # featuremap_visual(x4_feature)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3]


        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        # e4 = self.conv_reduces[3](x_fuses[3])
        # e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3]
        # print(torch.cat(outputs, dim=1).shape)

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs





class PiDiNet_1rfn(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False, img_name = None):
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
            for i in range(3):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(Attention(self.dil))
                # self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(3):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(3):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(3):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(3, 1, kernel_size=1) # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)


        # 我们需要在C5后面接一个1x1, 256 conv，得到金字塔最顶端的feature
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Smooth layers
        # 这个是上面引文中提到的抗aliasing的3x3卷积
        self.smooth1 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        # 为了匹配channel dimension引入的1x1卷积
        # 注意这些backbone之外的extra conv，输出都是256 channel
        self.latlayer1 = nn.Conv2d( 240, 240, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 120, 240, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 60,  240, kernel_size=1, stride=1, padding=0)

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
    
    ## FPN的lateral connection部分: upsample以后，element-wise相加
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x, x_name):
        # print('测试4——1')
        x_feature = x
        
        H, W = x.size()[2:]
        
        # print('H, W:',H, W)

        x = self.init_block(x)
        
        


        x1 = self.block1_1(x)
        
        

        x1 = self.block1_2(x1)
        

        x1 = self.block1_3(x1)
        
        
        x1_feature = x1
        # featuremap_visual(x1_feature)


        x2 = self.block2_1(x1)
        # featuremap_visual(x2)
        x2 = self.block2_2(x2)
        # featuremap_visual(x2)
        x2 = self.block2_3(x2)
        # featuremap_visual(x2)
        x2 = self.block2_4(x2)
        
        x2_feature = x2
        # featuremap_visual(x2_feature)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)
        
        x3_feature = x3
        # featuremap_visual(x3_feature)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)


        # P4: 上一层 p5 + 侧边来的 c4
        # 其余同理
        p3 = self._upsample_add(x4, self.latlayer1(x3))
        # p3_feature = p3
        # featuremap_visual(p3_feature,x_name)
        p2 = self._upsample_add(p3, self.latlayer2(x2))
        # p2_feature = p2
        # featuremap_visual(p2_feature,x_name)
        p1 = self._upsample_add(p2, self.latlayer3(x1))
        # p1_feature = p1
        # featuremap_visual(p1_feature,x_name)
        # Smooth
        # 输出做一下smooth
        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth3(p1)

        
        # x4_feature = x4
        # featuremap_visual(x4_feature)

        # x_fuses = []
        # if self.sa and self.dil is not None:
        #     for i, xi in enumerate([x1, x2, x3]):
        #         x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        # elif self.sa:
        #     for i, xi in enumerate([x1, x2, x3]):
        #         x_fuses.append(self.attentions[i](xi))
        # elif self.dil is not None:
        #     for i, xi in enumerate([x1, x2, x3]):
        #         x_fuses.append(self.dilations[i](xi))
        # else:
        #     x_fuses = [x1, x2, x3]

        p_fuses = []
        if self.sa and self.dil is not None:
            for i, pi in enumerate([p1, p2, p3]):
                p_fuses.append(self.attentions[i](self.dilations[i](pi)))
        elif self.sa:
            for i, pi in enumerate([p1, p2, p3]):
                p_fuses.append(self.attentions[i](pi))
        elif self.dil is not None:
            for i, pi in enumerate([x1, x2, x3]):
                p_fuses.append(self.dilations[i](pi))
        else:
            p_fuses = [p1, p2, p3]

        print(p_fuses[0].shape)
        print(p_fuses[1].shape)
        print(p_fuses[2].shape)

        e1 = self.conv_reduces[0](p_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](p_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](p_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        # e4 = self.conv_reduces[3](x_fuses[3])
        # e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3]
        # print(torch.cat(outputs, dim=1).shape)

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs



class PiDiNet_rfp(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False, img_name = None):
        super(PiDiNet_rfp, self).__init__()
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
            for i in range(3):
                self.dilations.append(CDCM_ASPP(self.fuseplanes[i], self.dil))
                self.attentions.append(Attention(self.dil))
                # self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(3):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(3):
                self.dilations.append(CDCM_ASPP(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(3):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
                
        self.fpn_block = nn.BatchNorm2d(nn.Conv2d())

        self.classifier = nn.Conv2d(3, 1, kernel_size=1) # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)


        # 我们需要在C5后面接一个1x1, 256 conv，得到金字塔最顶端的feature
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Smooth layers
        # 这个是上面引文中提到的抗aliasing的3x3卷积
        self.smooth1 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        # 为了匹配channel dimension引入的1x1卷积
        # 注意这些backbone之外的extra conv，输出都是256 channel
        self.latlayer1 = nn.Conv2d( 240, 240, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 120, 240, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 60,  240, kernel_size=1, stride=1, padding=0)


        #转换通道
        self.tranlayer1 = nn.Conv2d(24, 60, kernel_size=1, stride=1, padding=0)
        self.tranlayer2 = nn.Conv2d(120, 60, kernel_size=1, stride=1, padding=0)
        self.tranlayer3 = nn.Conv2d(144, 120, kernel_size=1, stride=1, padding=0)
        self.tranlayer4 = nn.Conv2d(24, 240, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(240, 24, kernel_size=1, padding=0)
        # self.x1_feature = nn.Conv2d(60, 1, kernel_size=1, padding=0)
        # self.x2_feature = nn.Conv2d(120, 1, kernel_size=1, padding=0)
        # self.x3_feature = nn.Conv2d(240, 1, kernel_size=1, padding=0)
        # self.x4_feature = nn.Conv2d(240, 1, kernel_size=1, padding=0)

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
    
    ## FPN的lateral connection部分: upsample以后，element-wise相加
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
    
    def _upsample(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear')

    def forward(self, x, x_name):
        # print('测试4——1')
        x_feature = x
        
        H, W = x.size()[2:]
        
        # print('H, W:',H, W)

        x = self.init_block(x)
        
        


        x1_old = self.block1_1(x)
        
        

        x1 = self.block1_2(x1_old)
        

        x1 = self.block1_3(x1)
        
        
        x1_feature = x1
        # featuremap_visual(x1_feature,x_name)


        x2_old = self.block2_1(x1)
        # featuremap_visual(x2)
        x2 = self.block2_2(x2_old)
        # featuremap_visual(x2)
        x2 = self.block2_3(x2)
        # featuremap_visual(x2)
        x2 = self.block2_4(x2)
        
        x2_feature = x2
        # featuremap_visual(x2_feature,x_name)

        x3_old = self.block3_1(x2)
        x3 = self.block3_2(x3_old)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)
        
        x3_feature = x3
        # featuremap_visual(x3_feature,x_name)

        x4_old = self.block4_1(x3)
        x4 = self.block4_2(x4_old)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)
        
        x4_feature = x4
        # featuremap_visual(x4_feature,x_name)


        # P4: 上一层 p5 + 侧边来的 c4
        # 其余同理
        p3 = self._upsample_add(x4, self.latlayer1(x3))
        # p3_feature = p3
        # featuremap_visual(p3_feature,x_name)
        p2 = self._upsample_add(p3, self.latlayer2(x2))
        # p2_feature = p2
        # featuremap_visual(p2_feature,x_name)
        p1 = self._upsample_add(p2, self.latlayer3(x1))
        # p1_feature = p1
        # featuremap_visual(p1_feature,x_name)
        # Smooth
        # 输出做一下smooth
        p3 = self.smooth1(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth3(p1)
 

        
        # x4_feature = x4
        # featuremap_visual(x4_feature)

        # x_fuses = []
        # if self.sa and self.dil is not None:
        #     for i, xi in enumerate([x1, x2, x3]):
        #         x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        # elif self.sa:
        #     for i, xi in enumerate([x1, x2, x3]):
        #         x_fuses.append(self.attentions[i](xi))
        # elif self.dil is not None:
        #     for i, xi in enumerate([x1, x2, x3]):
        #         x_fuses.append(self.dilations[i](xi))
        # else:
        #     x_fuses = [x1, x2, x3]

        p_fuses = []
        if self.sa and self.dil is not None:
            for i, pi in enumerate([p1, p2, p3]):
                p_fuses.append(self.attentions[i](self.dilations[i](pi)))
        elif self.sa:
            for i, pi in enumerate([p1, p2, p3]):
                p_fuses.append(self.attentions[i](pi))
        elif self.dil is not None:
            for i, pi in enumerate([p1, p2, p3]):
                p_fuses.append(self.dilations[i](pi))
        else:
            p_fuses = [p1, p2, p3]


        
        

        #空洞卷积后，第一层拼接, 并重新开始提取特征
        f1 = p_fuses[0]
        f1 = torch.cat((self.tranlayer1(self._upsample(f1,x1_old)), x1_old), axis =1)
        f1 = self.block1_2(self.tranlayer2(f1))
        f1 = self.block1_3(f1)



        f2 = p_fuses[1]
        f2 = torch.cat((self.tranlayer1(self._upsample(f2,f1)), f1), axis =1)
        f2 = self.block2_1(self.tranlayer2(f2))
        f2 = self.block2_2(f2)
        f2 = self.block2_3(f2)
        f2 = self.block2_4(f2)


        f3 = p_fuses[2]
        f3 = torch.cat((self._upsample(f3,f2), f2), axis = 1)      
        f3 = self.block3_1(self.tranlayer3(f3))
        f3 = self.block3_2(f3)
        f3 = self.block3_3(f3)
        f3 = self.block3_4(f3)

        #提取特征结束


        # Top-down
        # s4: 金字塔最顶上的feature
        s4 = self.relu(f3)
        # P4: 上一层 s4 + 侧边来的 f3
        # 其余同理
        s3 = self._upsample_add(s4, self.latlayer1(f3))
        s3 = self._upsample_add(s3,self.tranlayer4(p_fuses[2]))
        # p3_feature = p3
        # featuremap_visual(p3_feature,x_name)
        s2 = self._upsample_add(s3, self.latlayer2(f2))


        s2 = self._upsample_add(s2,self.tranlayer4(p_fuses[1]))
        # p2_feature = p2
        # featuremap_visual(p2_feature,x_name)
        s1 = self._upsample_add(s2, self.latlayer3(f1))
        s1 = self._upsample_add(s1,self.tranlayer4(p_fuses[0]))
        # p1_feature = p1
        # featuremap_visual(p1_feature,x_name)
        # Smooth
        # 输出做一下smooth
        s3 = self.conv_1(self.smooth1(s3))
        s2 = self.conv_1(self.smooth2(s2))
        s1 = self.conv_1(self.smooth3(s1))
        # s1_feature = s1
        # featuremap_visual(s1_feature, x_name)






        e1 = self.conv_reduces[0](s1)

        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](s2)
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](s3)
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        # e4 = self.conv_reduces[3](x_fuses[3])
        # e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3]
        # print(torch.cat(outputs, dim=1).shape)

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs



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
            for i in range(3):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(Attention(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(3):
                self.attentions.append(Attention(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(3):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(3):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(3, 1, kernel_size=1) # has bias
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

        # x4 = self.block4_1(x3)
        # x4 = self.block4_2(x4)
        # x4 = self.block4_3(x4)
        # x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        # e4 = self.conv_reduces[3](x_fuses[3])
        # e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3]

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


def pidinet_tiny(args):
    print('测试1——1')
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa)

def pidinet_small(args):
    print('测试1——2')
    pdcs = config_model(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa)

# def pidinet(args,img_name):
#     print('测试1——3')
#     pdcs = config_model(args.config)
#     dil = 24 if args.dil else None
#     return PiDiNet(60, pdcs, dil=dil, sa=args.sa, img_name = None)

def pidinet(args):
    pdcs = config_model(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa)


## convert pidinet to vanilla cnn

def pidinet_tiny_converted(args):
    print('测试1——4')
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_small_converted(args):
    print('测试1——5')
    pdcs = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_converted(args):
    print('测试1——6')
    pdcs = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa, convert=True)
