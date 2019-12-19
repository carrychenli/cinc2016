#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: main.py
# @Time    : 2019/12/18 16:21
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
from pcg_model import creat_pcg_model
import tensorflow as tf

register_matplotlib_converters()
plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False

import h5py
from datetime import datetime


def train_model(dim=1, x=None, y=None):
    print('Go model' + str(dim) + 'D!')

    log_dir = "logs\\model" + str(dim) + "D\\tsbd\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 在文件名中包含 epoch (使用 `str.format`)
    checkpoint_path = "logs\\model" + str(dim) + "D\\ckpt\\" + "cp-{epoch:04d}.ckpt"
    # 创建一个回调，每 5 个 epochs 保存模型的权重
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True, period=5)

    model = creat_pcg_model(dim, k[dim - 1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=x, y=y, batch_size=batch_size[dim - 1], epochs=epochs[dim - 1],
              shuffle=True, validation_split=0.1, verbose=2, callbacks=[tensorboard_callback, cp_callback])
    model.summary()
    model.evaluate(x=x, y=y, batch_size=batch_size[dim - 1])


if __name__ == '__main__':
    epochs = np.array([1, 1, 1]) * 2
    batch_size = [128, 128, 64]
    k = [1, 1, 1]
    with h5py.File('cincset1.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        # labels = h5f['label2d']
        labels = h5f['label1d']
        print(labels[:].shape)
        print(np.sum(labels[:]))

        train_model(dim=1, x=xs[:], y=labels[:])
        train_model(dim=2, x=specs[:], y=labels[:])
        train_model(dim=3, x=(xs[:], specs[:]), y=labels[:])
