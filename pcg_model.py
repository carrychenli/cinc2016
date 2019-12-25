#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: pcg_model.py
# @Time    : 2019/12/16 17:27
# @Description:
"""

"""

from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class BasicConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(BasicConv1D, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class Conv1DLinear(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(Conv1DLinear, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return x


class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(BasicConv2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class Conv2DLinear(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(Conv2DLinear, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return x


class Stem(tf.keras.layers.Layer):
    def __init__(self, dim=1, k=1):
        super(Stem, self).__init__()
        if dim == 1:
            self.conv1 = BasicConv1D(filters=1 * k, kernel_size=3, strides=2, padding="same")
            self.conv2 = BasicConv1D(filters=1 * k, kernel_size=3, strides=2, padding="same")
            self.conv3 = BasicConv1D(filters=2 * k, kernel_size=3, strides=2, padding="same")
            self.b1_maxpool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")
            self.b2_conv = BasicConv1D(filters=2 * k, kernel_size=3, strides=2, padding="same")
            self.b3_conv1 = BasicConv1D(filters=4 * k, kernel_size=1, strides=1, padding="same")
            self.b3_conv2 = BasicConv1D(filters=8 * k, kernel_size=3, strides=2, padding="valid")
            self.b4_conv1 = BasicConv1D(filters=4 * k, kernel_size=1, strides=1, padding="same")
            self.b4_conv2 = BasicConv1D(filters=4 * k, kernel_size=3, strides=1, padding="same")
            self.b4_conv3 = BasicConv1D(filters=4 * k, kernel_size=3, strides=1, padding="same")
            self.b4_conv4 = BasicConv1D(filters=8 * k, kernel_size=3, strides=2, padding="valid")
            self.b5_conv = BasicConv1D(filters=8 * k, kernel_size=3, strides=2, padding="same")
            self.b6_maxpool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")
        else:  # dim==2
            self.conv1 = BasicConv2D(filters=4 * k, kernel_size=3, strides=2, padding="same")
            self.conv2 = BasicConv2D(filters=4 * k, kernel_size=3, strides=1, padding="same")
            self.conv3 = BasicConv2D(filters=4 * k, kernel_size=3, strides=1, padding="same")
            self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            self.b2_conv = BasicConv2D(filters=4 * k, kernel_size=3, strides=2, padding="same")
            self.b3_conv1 = BasicConv2D(filters=8 * k, kernel_size=1, strides=1, padding="same")
            self.b3_conv2 = BasicConv2D(filters=8 * k, kernel_size=3, strides=1, padding="same")
            self.b4_conv1 = BasicConv2D(filters=8 * k, kernel_size=1, strides=1, padding="same")
            self.b4_conv2 = BasicConv2D(filters=8 * k, kernel_size=3, strides=1, padding="same")
            self.b4_conv3 = BasicConv2D(filters=8 * k, kernel_size=3, strides=1, padding="same")
            self.b4_conv4 = BasicConv2D(filters=8 * k, kernel_size=3, strides=1, padding="same")
            self.b5_conv = BasicConv2D(filters=8 * k, kernel_size=3, strides=2, padding="same")
            self.b6_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")

    def call(self, inputs, training=None, **kwargs):  # 128*128*1 for 2D,  # 10000*1 for 1D
        x = self.conv1(inputs, training=training)  # 64*64*4k for 2D,  # 5000*1k for 1D
        x = self.conv2(x, training=training)  # 64*64*4k for 2D,  # 2500*1k for 1D
        x = self.conv3(x, training=training)  # 64*64*4k for 2D,  # 1250*2k for 1D
        branch_1 = self.b1_maxpool(x)  # 32*32*4k for 2D,  # 625*2k for 1D
        branch_2 = self.b2_conv(x, training=training)  # 32*32*4k for 2D,  # 625*2k for 1D
        x = tf.concat(values=[branch_1, branch_2], axis=-1)  # 32*32*8k for 2D,  # 625*4k for 1D
        branch_3 = self.b3_conv1(x, training=training)  # 32*32*8k for 2D,  # 625*4k for 1D
        branch_3 = self.b3_conv2(branch_3, training=training)  # 32*32*8k for 2D,  # 312*8k for 1D
        branch_4 = self.b4_conv1(x, training=training)  # 32*32*8k for 2D,  # 625*4k for 1D
        branch_4 = self.b4_conv2(branch_4, training=training)  # 32*32*8k for 2D,  # 625*4k for 1D
        branch_4 = self.b4_conv3(branch_4, training=training)  # 32*32*8k for 2D,  # 625*4k for 1D
        branch_4 = self.b4_conv4(branch_4, training=training)  # 32*32*8k for 2D,  # 312*8k for 1D
        x = tf.concat(values=[branch_3, branch_4], axis=-1)  # 32*32*16k for 2D,  # 312*16k for 1D
        branch_5 = self.b5_conv(x, training=training)  # 16*16*8k for 2D,  # 312*8k for 1D
        branch_6 = self.b6_maxpool(x, training=training)  # 16*16*16k for 2D,  # 312*16k for 1D
        x = tf.concat(values=[branch_5, branch_6], axis=-1)  # 16*16*24k for 2D  # 156*24k for 1D
        return x


class InceptionResnetA(tf.keras.layers.Layer):
    def __init__(self, dim=1, k=1):
        super(InceptionResnetA, self).__init__()
        if dim == 1:
            self.BasicConvND = BasicConv1D
            self.ConvNDLinear = Conv1DLinear
        else:
            self.BasicConvND = BasicConv2D
            self.ConvNDLinear = Conv2DLinear
        self.conv1_1 = self.BasicConvND(8 * k, 1, strides=1, padding='same')
        self.conv2_1 = self.BasicConvND(8 * k, 1, strides=1, padding='same')
        self.conv2_2 = self.BasicConvND(8 * k, 3, strides=1, padding='same')
        self.conv3_1 = self.BasicConvND(8 * k, 1, strides=1, padding='same')
        self.conv3_2 = self.BasicConvND(8 * k, 3, strides=1, padding='same')
        self.conv3_3 = self.BasicConvND(8 * k, 3, strides=1, padding='same')
        self.conv = self.ConvNDLinear(24 * k, 1, strides=1, padding='same')

    def call(self, inputs, training=None, **kwargs):  # 16*16*24k for 2D  # 156*24k for 1D
        b1 = self.conv1_1(inputs, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        b2 = self.conv2_1(inputs, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        b2 = self.conv2_2(b2, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        b3 = self.conv3_1(inputs, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        b3 = self.conv3_2(b3, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        b3 = self.conv3_3(b3, training=training)  # 16*16*24k for 2D  # 156*24k for 1D

        x = tf.concat([b1, b2, b3], axis=-1)  # 16*16*72k for 2D  # 156*72k for 1D
        x = self.conv(x, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        x = tf.keras.layers.add([x, inputs])  # 16*16*24k for 2D  # 156*24k for 1D
        return tf.nn.relu(x)


class ReductionA(tf.keras.layers.Layer):
    def __init__(self, dim=1, k=1):
        super(ReductionA, self).__init__()
        if dim == 1:
            self.b1_pool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")
            self.b2_conv = BasicConv1D(filters=12 * k, kernel_size=3, strides=2, padding="same")
            self.b3_conv1 = BasicConv1D(filters=12 * k, kernel_size=1, strides=1, padding="same")
            self.b3_conv2 = BasicConv1D(filters=12 * k, kernel_size=3, strides=1, padding="same")
            self.b3_conv3 = BasicConv1D(filters=12 * k, kernel_size=3, strides=2, padding="same")
            self.conv = Conv1DLinear(filters=24 * k, kernel_size=1, strides=1, padding="same")
        else:
            self.b1_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            self.b2_conv = BasicConv2D(filters=12 * k, kernel_size=3, strides=2, padding="same")
            self.b3_conv1 = BasicConv2D(filters=12 * k, kernel_size=1, strides=1, padding="same")
            self.b3_conv2 = BasicConv2D(filters=12 * k, kernel_size=3, strides=1, padding="same")
            self.b3_conv3 = BasicConv2D(filters=12 * k, kernel_size=3, strides=2, padding="same")
            self.conv = Conv2DLinear(filters=24 * k, kernel_size=1, strides=1, padding="same")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_pool(inputs)
        b2 = self.b2_conv(inputs, training=training)
        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)
        b3 = self.b3_conv3(b3, training=training)
        x = tf.concat(values=[b1, b2, b3], axis=-1)
        return self.conv(x, training=training)


def build_inception_resnet_a(n, dim, k=1):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(InceptionResnetA(dim=dim, k=k))
    return block


class InceptionResnetV2(tf.keras.layers.Layer):
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
            self.avgpool = tf.keras.layers.AveragePooling1D(pool_size=10)
        else:
            self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=4)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):  # 128*128*1 for 2D  # 10000*1 for 1D
        x = self.stem(inputs, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        x = self.inception_resnet_a1(x, training=training)  # 16*16*24k for 2D  # 156*24k for 1D
        x = self.reduction_a1(x, training=training)  # 8*8*24k for 2D, # 78*24k for 1D
        if self.dim == 1:
            x = self.reduction_a1_more(x, training=training)  # 39*24k for 1D
        x = self.inception_resnet_a2(x, training=training)  # 8*8*24k for 2D, 39*24k for 1D
        x = self.reduction_a2(x, training=training)  # 4*4*24k for 2D, # 20*24k for 1D
        if self.dim == 1:
            x = self.reduction_a2_more(x, training=training)  # 10*24k for 1D
        x = self.avgpool(x)  # 1*1*24k for 2D, # 1*24k 1D
        x = self.dropout(x, training=training)  # 1*1*24k for 2D, # 1*24k 1D
        x = self.flat(x)  # 24k for 2D or 1D
        return x


class PCG(tf.keras.Model):
    def __init__(self, k=1, mode='add'):
        super(PCG, self).__init__()
        self.InceptionResnetV2D1 = InceptionResnetV2(dim=1, k=k)
        self.InceptionResnetV2D2 = InceptionResnetV2(dim=2, k=k)
        self.fc0 = tf.keras.layers.Dense(units=24, activation='relu')
        self.fc = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
        self.mode = mode

    def call(self, inputs, training=None, **kwargs):
        x1 = self.InceptionResnetV2D1(inputs[0])
        x2 = self.InceptionResnetV2D2(inputs[1])
        if self.mode == 'add':
            x = layers.add([x1, x2])
        else:  # self.mode =='concat':
            x = tf.concat(values=[x1, x2], axis=-1)
        x = self.fc0(x)
        return self.fc(x)


class PCGONLY(tf.keras.Model):
    def __init__(self, dim=1, k=1):
        super(PCGONLY, self).__init__()
        assert dim in [1, 2]
        self.InceptionResnetV2 = InceptionResnetV2(dim=dim, k=k)
        self.fc0 = tf.keras.layers.Dense(units=24, activation='relu')
        self.fc = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, **kwargs):
        x = self.InceptionResnetV2(inputs)
        x = self.fc0(x)
        return self.fc(x)


def creat_pcg_model(dim=1, k=1):
    assert dim in [1, 2, 3]  # 一定只有这3种情况，不能改
    assert k in range(1, 10)  # 不想模型太大或者太小，若一定要增加模型规模，可改大
    if dim in [1, 2]:
        return PCGONLY(dim=dim, k=k)
    # mode = 'add'
    mode = 'concat'
    return PCG(k=k, mode=mode)
