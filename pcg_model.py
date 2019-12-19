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
from inception_modules import BasicConv1D, BasicConv2D, Conv1DLinear, Conv2DLinear


class Stem(tf.keras.layers.Layer):
    def __init__(self, dim=1):
        super(Stem, self).__init__()
        if dim == 1:
            self.conv1 = BasicConv1D(filters=4, kernel_size=3, strides=2, padding="same")  # 5000*4
            self.conv2 = BasicConv1D(filters=4, kernel_size=3, strides=2, padding="same")  # 2500*4
            self.conv3 = BasicConv1D(filters=4, kernel_size=3, strides=2, padding="same")  # 1250*4
            self.b1_maxpool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")  # 625*4
            self.b2_conv = BasicConv1D(filters=4, kernel_size=3, strides=2, padding="same")  # 625*4
            self.b3_conv1 = BasicConv1D(filters=8, kernel_size=1, strides=1, padding="same")  # 625*8
            self.b3_conv2 = BasicConv1D(filters=8, kernel_size=3, strides=2, padding="valid")  # 312*8
            self.b4_conv1 = BasicConv1D(filters=8, kernel_size=1, strides=1, padding="same")  # 625*8
            self.b4_conv2 = BasicConv1D(filters=8, kernel_size=3, strides=1, padding="same")  # 625*8
            self.b4_conv3 = BasicConv1D(filters=8, kernel_size=3, strides=1, padding="same")  # 625*8
            self.b4_conv4 = BasicConv1D(filters=8, kernel_size=3, strides=2, padding="valid")  # 312*8
            self.b5_conv = BasicConv1D(filters=8, kernel_size=3, strides=2, padding="same")  # 156*8
            self.b6_maxpool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")  # 156*8
        else:  # dim==2
            self.conv1 = BasicConv2D(filters=4, kernel_size=3, strides=2, padding="same")  # 64*64*4
            self.conv2 = BasicConv2D(filters=4, kernel_size=3, strides=1, padding="same")  # 64*64*4
            self.conv3 = BasicConv2D(filters=4, kernel_size=3, strides=1, padding="same")  # 64*64*4
            self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")  # 32*32*4
            self.b2_conv = BasicConv2D(filters=4, kernel_size=3, strides=2, padding="same")  # 32*32*4
            self.b3_conv1 = BasicConv2D(filters=8, kernel_size=1, strides=1, padding="same")  # 32*32*8
            self.b3_conv2 = BasicConv2D(filters=8, kernel_size=3, strides=1, padding="same")  # 32*32*8
            self.b4_conv1 = BasicConv2D(filters=8, kernel_size=1, strides=1, padding="same")  # 32*32*64
            self.b4_conv2 = BasicConv2D(filters=8, kernel_size=3, strides=1, padding="same")  # 32*32*64
            self.b4_conv3 = BasicConv2D(filters=8, kernel_size=3, strides=1, padding="same")  # 32*32*64
            self.b4_conv4 = BasicConv2D(filters=8, kernel_size=3, strides=1, padding="same")  # 32*32*64
            self.b5_conv = BasicConv2D(filters=8, kernel_size=3, strides=2, padding="same")  # 16*16*64
            self.b6_maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")  # 16*16*64

    def call(self, inputs, training=None, **kwargs):  # 16*16*24 for 2D  # 156*24 for 1D
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        branch_1 = self.b1_maxpool(x)
        branch_2 = self.b2_conv(x, training=training)
        x = tf.concat(values=[branch_1, branch_2], axis=-1)
        branch_3 = self.b3_conv1(x, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_4 = self.b4_conv1(x, training=training)
        branch_4 = self.b4_conv2(branch_4, training=training)
        branch_4 = self.b4_conv3(branch_4, training=training)
        branch_4 = self.b4_conv4(branch_4, training=training)
        x = tf.concat(values=[branch_3, branch_4], axis=-1)
        branch_5 = self.b5_conv(x, training=training)
        branch_6 = self.b6_maxpool(x, training=training)
        x = tf.concat(values=[branch_5, branch_6], axis=-1)
        # print('tell me why ', x.shape, branch_5.shape, branch_6.shape)
        # exit()
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
        b1 = self.conv1_1(inputs, training=training)
        b2 = self.conv2_1(inputs, training=training)
        b2 = self.conv2_2(b2, training=training)
        b3 = self.conv3_1(inputs, training=training)
        b3 = self.conv3_2(b3, training=training)
        b3 = self.conv3_3(b3, training=training)

        x = tf.concat([b1, b2, b3], axis=-1)
        x = self.conv(x, training=training)
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
            self.conv = BasicConv1D(filters=24 * k, kernel_size=1, strides=1, padding="same")
        else:
            self.b1_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
            self.b2_conv = BasicConv2D(filters=12 * k, kernel_size=3, strides=2, padding="same")
            self.b3_conv1 = BasicConv2D(filters=12 * k, kernel_size=1, strides=1, padding="same")
            self.b3_conv2 = BasicConv2D(filters=12 * k, kernel_size=3, strides=1, padding="same")
            self.b3_conv3 = BasicConv2D(filters=12 * k, kernel_size=3, strides=2, padding="same")
            self.conv = BasicConv2D(filters=24 * k, kernel_size=1, strides=1, padding="same")

    def call(self, inputs, training=None, **kwargs):  # 8*8*48k for 2D, # 78*78*48k
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
    def __init__(self, dim=1):
        super(InceptionResnetV2, self).__init__()
        self.dim = dim
        self.stem = Stem(dim=dim)  # 16*16*24 for 2D  # 156*24for 1D
        self.inception_resnet_a1 = build_inception_resnet_a(3, dim=dim, k=1)  # 16*16*16 for 2D  # 156*16 for 1D
        self.reduction_a1 = ReductionA(dim=dim, k=1)  # 8*8*24 for 2D, # 78*24 for 1D
        self.inception_resnet_a2 = build_inception_resnet_a(3, dim=dim, k=1)
        self.reduction_a2 = ReductionA(dim=dim, k=1)  # 4*4*24 for 2D, # 19*24 for 1D
        if dim == 1:
            self.avgpool = tf.keras.layers.AveragePooling1D(pool_size=9)
            self.reduction_a1_more = ReductionA(dim=dim, k=1)  # 39*24 for 1D
            self.reduction_a2_more = ReductionA(dim=dim, k=1)  # 9*24 for 1D
        else:
            self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=4)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.flat = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs, training=training)
        x = self.inception_resnet_a1(x, training=training)
        x = self.reduction_a1(x, training=training)
        if self.dim == 1:
            x = self.reduction_a1_more(x, training=training)
        x = self.inception_resnet_a2(x, training=training)
        x = self.reduction_a2(x, training=training)
        if self.dim == 1:
            x = self.reduction_a2_more(x, training=training)
        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.flat(x)
        return x


class PCG(tf.keras.Model):
    def __init__(self):
        super(PCG, self).__init__()
        self.InceptionResnetV2D1 = InceptionResnetV2(dim=1)
        self.InceptionResnetV2D2 = InceptionResnetV2(dim=2)
        self.fc = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, **kwargs):
        x1 = self.InceptionResnetV2D1(inputs[0])
        x2 = self.InceptionResnetV2D2(inputs[1])
        x = layers.add([x1, x2])
        return self.fc(x)


class PCGONLY(tf.keras.Model):
    def __init__(self, dim=1):
        super(PCGONLY, self).__init__()
        assert dim in [1, 2]
        self.InceptionResnetV2 = InceptionResnetV2(dim=dim)
        self.fc = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, **kwargs):
        x = self.InceptionResnetV2(inputs)
        return self.fc(x)
