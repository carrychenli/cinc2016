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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
register_matplotlib_converters()
plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False

import h5py
import pickle
from datetime import datetime

from common import eval_model_common, generate_dataset, generate_dataset_3D


def train_model_h5py_generate(dim=1, x=None, y=None):
    print('Go to train model' + str(dim) + 'D!')

    log_dir = "logs\\model" + str(dim) + "D\\tsbd\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tsbd_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 在文件名中包含 epoch (使用 `str.format`)
    checkpoint_path = "logs\\model" + str(dim) + "D\\ckpt\\" + "cp-{epoch:04d}.ckpt"
    # 创建一个回调，每 5 个 epochs 保存模型的权重
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True, period=1)
    model = creat_pcg_model(dim, k[dim - 1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    g_fn = generate_dataset if dim in [1, 2] else generate_dataset_3D()
    args_train = [x, y] if dim in [1, 2] else [x[0], x[1], y]
    args_val = args_train + ['val']
    print('position 1')
    train_set = tf.data.Dataset.from_generator(generator=g_fn, args=args_train,
                                               output_types=output_types[dim - 1],
                                               output_shapes=output_shapes[dim - 1])
    print('position 2')  # 这里还是会内存占用而卡机
    val_set = tf.data.Dataset.from_generator(generator=g_fn, args=args_val,
                                             output_types=output_types[dim - 1],
                                             output_shapes=output_shapes[dim - 1])
    print('position 3')
    train_set = train_set.batch(batch[dim - 1]).repeat()
    print('position 4')
    val_set = val_set.batch(batch[dim - 1]).repeat()
    print('position 5')

    model.fit(train_set, epochs=epochs[dim - 1], shuffle=False, verbose=1, validation_data=val_set,
              callbacks=[tsbd_callback, ckpt_callback])
    model.summary()
    return None


def train_model_small(dim=1, x=None, y=None):
    print('Go to train model' + str(dim) + 'D!')

    log_dir = "logs\\model" + str(dim) + "D\\tsbd\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tsbd_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 在文件名中包含 epoch (使用 `str.format`)
    checkpoint_path = "logs\\model" + str(dim) + "D\\ckpt\\" + "cp-{epoch:04d}.ckpt"
    # 创建一个回调，每 5 个 epochs 保存模型的权重
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True, period=1)

    model = creat_pcg_model(dim, k[dim - 1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=x, y=y, batch_size=batch[dim - 1], epochs=epochs[dim - 1],
              shuffle=True, validation_split=0.1, verbose=2, callbacks=[tsbd_callback, ckpt_callback])
    model.summary()
    model.evaluate(x=x, y=y, batch_size=batch[dim - 1])
    # model.save("logs\\model" + str(dim) + "D.h5", save_format='tf')
    return None


def train_model_h5py(dim=1, x=None, y=None):
    print('Go to train model' + str(dim) + 'D!')

    log_dir = "logs\\model" + str(dim) + "D\\tsbd\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tsbd_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 在文件名中包含 epoch (使用 `str.format`)
    ckpt_path = "logs\\model" + str(dim) + "D\\ckpt\\" + "cp.ckpt"
    # 创建一个回调，每 5 个 epochs 保存模型的权重
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path, verbose=1, save_weights_only=True, period=1)
    model = creat_pcg_model(dim, k[dim - 1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    bg, ed = 0, 1280
    for i in range(epochs[dim - 1]):
        while ed < y.shape[0]:
            print("model%dD. Epoch:%d/%d, batch: %d/%d" % (dim, i + 1, epochs[dim - 1], ed / 1280, y.shape[0] // 1280))
            xi = x[bg:ed] if dim in [1, 2] else (x[0][bg:ed], x[1][bg:ed])
            yi = y[bg:ed]
            model.fit(x=xi, y=yi, batch_size=batch[dim - 1], epochs=1,
                      shuffle=True, validation_split=0.1, verbose=2, callbacks=[tsbd_callback, ckpt_callback])
            bg = ed
            ed += 1280
    model.summary()
    return None


def eval_model(dim=1, x=None, y=None):
    print('Go to evaluate model' + str(dim) + 'D!')
    model = creat_pcg_model(dim, k[dim - 1])
    latest = tf.train.latest_checkpoint("logs\\model" + str(dim) + "D\\ckpt")
    model.load_weights(latest)
    eval_model_common(model, dim, x, y)
    return None


output_types = [(tf.float32, tf.int64),
                (tf.float32, tf.int64),
                ((tf.float32, tf.float32), tf.int64)]
output_shapes = [((10000, 1), ()),
                 ((128, 128, 1), ()),
                 (((10000, 1), (128, 128, 1)), ())]
epochs = np.array([1, 1, 1]) * 2
batch = [32, 32, 32]
k = np.array([1, 1, 1]) * 8

# train h5py
if __name__ == '__main__':
    with h5py.File('cincset1.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        labels = h5f['label1d']  # labels = h5f['label2d']
        #
        train_model_h5py_generate(dim=1, x=xs, y=labels)
        # train_model_h5py(dim=2, x=specs, y=labels)
        # train_model_h5py(dim=3, x=(xs, specs), y=labels)

# train small
if __name__ == '__main__A':
    with h5py.File('cincset1.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        labels = h5f['label1d']  # labels = h5f['label2d']
        # print(labels[:].shape)
        # print(np.sum(labels[:]))
        #
        train_model_small(dim=1, x=xs[:], y=labels[:])
        train_model_small(dim=2, x=specs[:], y=labels[:])
        train_model_small(dim=3, x=(xs[:], specs[:]), y=labels[:])

# evaluate
if __name__ == '__main__':
    with open('val.pkl', 'rb') as f:
        d = pickle.load(f)
        # x, spec, y = \  # for train data
        #     d['x_train_for_val'], d['spec_train_for_val'], d['y_train_for_val']
        x, spec, y = d['x_val'], d['spec_val'], d['y_val']
        eval_model(dim=1, x=x, y=y)
        # eval_model(dim=2, x=spec, y=y)
        # eval_model(dim=3, x=(x, spec), y=y)

    # with h5py.File('val.h5', 'r') as h5f:
    #     x_val = h5f['x_val']
    #     spec_val = h5f['spec_val']
    #     y_val = h5f['y_val']
    #     x_train_for_val = h5f['x_train_for_val']
    #     spec_train_for_val = h5f['spec_train_for_val']
    #     y_train_for_val = h5f['y_train_for_val']
    #     eval_model(dim=1, x=x_val, y=y_val)
    #     eval_model(dim=2, x=spec_val, y=y_val)
    #     eval_model(dim=3, x=(x_val, spec_val), y=y_val)
