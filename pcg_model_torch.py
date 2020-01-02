#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: pcg_model_torch.py
# @Time    : 2019/12/31 14:18
# @Description:
"""

"""

from ndtpy.tools import list_all_files, get_fig_title, maximize_figure

from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, inc, outc, ks, sd, pad, dim=1):
        super(BasicConv, self).__init__()
        if dim == 1:
            self.conv = nn.Conv1d(inc, outc, ks, sd, pad)
            self.bn = nn.BatchNorm1d(outc, eps=0.0001)
        else:  # dim==2
            self.conv = nn.Conv2d(inc, outc, ks, sd, pad)
            self.bn = nn.BatchNorm2d(outc, eps=0.0001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class BasicConvLinear(nn.Module):
    def __init__(self, inc, outc, ks, sd, pad, dim=1):
        super(BasicConvLinear, self).__init__()
        if dim == 1:
            self.conv = nn.Conv1d(inc, outc, ks, sd, pad)
            self.bn = nn.BatchNorm1d(outc)
        else:  # dim==2
            self.conv = nn.Conv2d(inc, outc, ks, sd, pad)
            self.bn = nn.BatchNorm2d(outc)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Stem(nn.Module):
    def __init__(self, dim=1, k=1):
        super(Stem, self).__init__()
        if dim == 1:
            self.conv1 = BasicConv(1, 1 * k, 3, 2, 1)
            self.conv2 = BasicConv(1 * k, 1 * k, 3, 2, 1)
            self.conv3 = BasicConv(1 * k, 2 * k, 3, 2, 1)
            self.b1_maxpool = nn.MaxPool1d(3, 2, 1)
            self.b2_conv = BasicConv(2 * k, 2 * k, 3, 2, 1)
            self.b3_conv1 = BasicConv(4 * k, 4 * k, 1, 1, 0)
            self.b3_conv2 = BasicConv(4 * k, 8 * k, 3, 2, 0)  # valid
            self.b4_conv1 = BasicConv(4 * k, 4 * k, 1, 1, 0)
            self.b4_conv2 = BasicConv(4 * k, 4 * k, 3, 1, 1)
            self.b4_conv3 = BasicConv(4 * k, 4 * k, 3, 1, 1)
            self.b4_conv4 = BasicConv(4 * k, 8 * k, 3, 2, 0)  # valid
            self.b5_conv = BasicConv(16 * k, 8 * k, 3, 2, 1)
            self.b6_maxpool = nn.MaxPool1d(3, 2, 1)
        else:  # dim == 2:
            self.conv1 = BasicConv(1, 4 * k, 3, 2, 1, 2)
            self.conv2 = BasicConv(4 * k, 4 * k, 3, 1, 1, 2)
            self.conv3 = BasicConv(4 * k, 4 * k, 3, 1, 1, 2)
            self.b1_maxpool = nn.MaxPool2d(3, 2, 1)
            self.b2_conv = BasicConv(4 * k, 4 * k, 3, 2, 1, 2)
            self.b3_conv1 = BasicConv(8 * k, 8 * k, 1, 1, 0, 2)
            self.b3_conv2 = BasicConv(8 * k, 8 * k, 3, 1, 1, 2)
            self.b4_conv1 = BasicConv(8 * k, 8 * k, 1, 1, 0, 2)
            self.b4_conv2 = BasicConv(8 * k, 8 * k, 3, 1, 1, 2)
            self.b4_conv3 = BasicConv(8 * k, 8 * k, 3, 1, 1, 2)
            self.b4_conv4 = BasicConv(8 * k, 8 * k, 3, 1, 1, 2)
            self.b5_conv = BasicConv(16 * k, 8 * k, 3, 2, 1, 2)
            self.b6_maxpool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        branch_1 = self.b1_maxpool(x)
        branch_2 = self.b2_conv(x)
        x = torch.cat([branch_1, branch_2], dim=1)
        branch_3 = self.b3_conv1(x)
        branch_3 = self.b3_conv2(branch_3)
        branch_4 = self.b4_conv1(x)
        branch_4 = self.b4_conv2(branch_4)
        branch_4 = self.b4_conv3(branch_4)
        branch_4 = self.b4_conv4(branch_4)
        x = torch.cat([branch_3, branch_4], dim=1)
        branch_5 = self.b5_conv(x)
        branch_6 = self.b6_maxpool(x)
        x = torch.cat([branch_5, branch_6], dim=1)  # 24 * 16 * 16 for 2D
        # print('stem:', x.shape)
        return x


class InceptionResnetA(nn.Module):
    def __init__(self, dim=1, k=1):
        super(InceptionResnetA, self).__init__()
        self.conv1_1 = BasicConv(24 * k, 8 * k, 1, 1, 0, dim)
        self.conv2_1 = BasicConv(24 * k, 8 * k, 1, 1, 0, dim)
        self.conv2_2 = BasicConv(8 * k, 8 * k, 3, 1, 1, dim)
        self.conv3_1 = BasicConv(24 * k, 8 * k, 1, 1, 0, dim)
        self.conv3_2 = BasicConv(8 * k, 8 * k, 3, 1, 1, dim)
        self.conv3_3 = BasicConv(8 * k, 8 * k, 3, 1, 1, dim)
        self.conv = BasicConvLinear(24 * k, 24 * k, 3, 1, 1, dim)

    def forward(self, inputs):  # 16*16*24k for 2D  # 156*24k for 1D
        b1 = self.conv1_1(inputs)  # 16*16*24k for 2D  # 156*24k for 1D
        b2 = self.conv2_1(inputs)  # 16*16*24k for 2D  # 156*24k for 1D
        b2 = self.conv2_2(b2)  # 16*16*24k for 2D  # 156*24k for 1D
        b3 = self.conv3_1(inputs)  # 16*16*24k for 2D  # 156*24k for 1D
        b3 = self.conv3_2(b3)  # 16*16*24k for 2D  # 156*24k for 1D
        b3 = self.conv3_3(b3)  # 16*16*24k for 2D  # 156*24k for 1D

        x = torch.cat([b1, b2, b3], dim=1)  # 16*16*72k for 2D  # 156*72k for 1D
        x = self.conv(x)  # 16*16*24k for 2D  # 156*24k for 1D
        x = x + inputs  # 16*16*24k for 2D  # 156*24k for 1D
        # print('InceptionResnetA:', x.shape)
        return F.relu(x, inplace=True)


class ReductionA(nn.Module):
    def __init__(self, dim=1, k=1):
        super(ReductionA, self).__init__()
        if dim == 1:
            self.b1_pool = nn.MaxPool1d(3, 2, 1)
        else:
            self.b1_pool = nn.MaxPool2d(3, 2, 1)
        self.b2_conv = BasicConv(24 * k, 12 * k, 3, 2, 1, dim)
        self.b3_conv1 = BasicConv(24 * k, 12 * k, 1, 1, 0, dim)
        self.b3_conv2 = BasicConv(12 * k, 12 * k, 3, 1, 1, dim)
        self.b3_conv3 = BasicConv(12 * k, 12 * k, 3, 2, 1, dim)
        self.conv = BasicConvLinear(48 * k, 24 * k, 1, 1, 0, dim)

    def forward(self, inputs):
        b1 = self.b1_pool(inputs)
        b2 = self.b2_conv(inputs)
        b3 = self.b3_conv1(inputs)
        b3 = self.b3_conv2(b3)
        b3 = self.b3_conv3(b3)
        x = torch.cat([b1, b2, b3], dim=1)
        # print('ReductionA:', x.shape)
        return self.conv(x)


def build_inception_resnet_a(n, dim, k=1):
    block = nn.Sequential()
    for nid in range(n):
        block.add_module('InceptionA' + str(nid + 1), InceptionResnetA(dim=dim, k=k))
    return block


class InceptionResnetV2(nn.Module):
    def __init__(self, dim=1, k=1):
        super(InceptionResnetV2, self).__init__()
        self.dim = dim
        self.stem = Stem(dim=dim, k=k)
        self.inception_resnet_a1 = build_inception_resnet_a(3, dim=dim, k=k)
        self.reduction_a1 = ReductionA(dim=dim, k=k)
        self.inception_resnet_a2 = build_inception_resnet_a(3, dim=dim, k=k)
        self.reduction_a2 = ReductionA(dim=dim, k=k)
        if dim == 1:
            self.reduction_a1_more = ReductionA(dim=dim, k=k)
            self.reduction_a2_more = ReductionA(dim=dim, k=k)
            self.avgpool = nn.AvgPool1d(10)
        else:
            self.avgpool = nn.AvgPool2d(10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):  # 128*128*1 for 2D  # 10000*1 for 1D
        x = self.stem(inputs)  # 16*16*24k for 2D  # 156*24k for 1D
        x = self.inception_resnet_a1(x)  # 16*16*24k for 2D  # 156*24k for 1D
        x = self.reduction_a1(x)  # 8*8*24k for 2D, # 78*24k for 1D
        if self.dim == 1:
            x = self.reduction_a1_more(x)  # 39*24k for 1D
        x = self.inception_resnet_a2(x)  # 8*8*24k for 2D, 39*24k for 1D
        x = self.reduction_a2(x)  # 4*4*24k for 2D, # 20*24k for 1D
        if self.dim == 1:
            x = self.reduction_a2_more(x)  # 10*24k for 1D
        x = self.avgpool(x)  # 1*1*24k for 2D, # 1*24k 1D
        x = self.dropout(x)  # 1*1*24k for 2D, # 1*24k 1D
        x = x.view(x.size(0), -1)
        return x


class PCG(nn.Module):
    def __init__(self, k=1):
        super(PCG, self).__init__()
        self.InceptionResnetV2D1 = InceptionResnetV2(dim=1, k=k)
        self.InceptionResnetV2D2 = InceptionResnetV2(dim=2, k=k)
        self.fc0 = nn.Linear(48 * k, 24)
        self.fc = nn.Linear(24, 1)

    def forward(self, inputs):
        x1 = self.InceptionResnetV2D1(inputs[0])
        x2 = self.InceptionResnetV2D2(inputs[1])
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc0(x))
        return torch.sigmoid(self.fc(x))


class PCGONLY(nn.Module):
    def __init__(self, dim=1, k=1):
        super(PCGONLY, self).__init__()
        assert dim in [1, 2]
        self.InceptionResnetV2 = InceptionResnetV2(dim=dim, k=k)
        self.fc0 = nn.Linear(24 * k, 24)
        self.fc = nn.Linear(24, 1)

    def forward(self, inputs):
        x = self.InceptionResnetV2(inputs)
        x = F.relu(self.fc0(x))
        return torch.sigmoid(self.fc(x))


def create_pcg_model(dim, k):
    assert dim in [1, 2, 3]  # 一定只有这3种情况，不能改
    assert k in range(1, 10)  # 不想模型太大或者太小，若一定要增加模型规模，可改大
    if dim in [1, 2]:
        return PCGONLY(dim=dim, k=k)
    return PCG(k=k)
