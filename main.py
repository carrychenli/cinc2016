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


def generate_train_dataset(x, y):
    i = 0
    while i < y.shape[0] * 0.9:
        data = (x[i], y[i])
        i += 1
        yield data


def generate_val_dataset(x, y):
    i = int(y.shape[0] * 0.9)
    while i < y.shape[0]:
        data = (x[i], y[i])
        i += 1
        yield data


def generate_train_dataset_3D(x0, x1, y):
    i = 0
    while i < y.shape[0] * 0.9:
        data = ((x0[i], x1[i]), y[i])
        i += 1
        yield data


def generate_val_dataset_3D(x0, x1, y):
    i = int(y.shape[0] * 0.9)
    while i < y.shape[0]:
        data = ((x0[i], x1[i]), y[i])
        i += 1
        yield data


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

    if dim in [1, 2]:
        train_set = tf.data.Dataset.from_generator(generator=generate_train_dataset, args=[x, y],
                                                   output_types=output_types[dim - 1],
                                                   output_shapes=output_shapes[dim - 1])
        val_set = tf.data.Dataset.from_generator(generator=generate_val_dataset, args=[x, y],
                                                 output_types=output_types[dim - 1],
                                                 output_shapes=output_shapes[dim - 1])
    else:
        train_set = tf.data.Dataset.from_generator(generator=generate_train_dataset_3D, args=[x[0], x[1], y],
                                                   output_types=output_types[dim - 1],
                                                   output_shapes=output_shapes[dim - 1])
        val_set = tf.data.Dataset.from_generator(generator=generate_val_dataset_3D, args=[x[0], x[1], y],
                                                 output_types=output_types[dim - 1],
                                                 output_shapes=output_shapes[dim - 1])

    train_set = train_set.batch(batch[dim - 1])
    val_set = val_set.batch(batch[dim - 1])

    model.fit(train_set, epochs=1, shuffle=False, verbose=2, validation_data=val_set,
              callbacks=[tsbd_callback, ckpt_callback])  # validation_data=val_set,
    model.summary()
    # model.save("logs\\model" + str(dim) + "D.h5", save_format='tf')
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
    checkpoint_path = "logs\\model" + str(dim) + "D\\ckpt\\" + "cp-{epoch:04d}.ckpt"
    # 创建一个回调，每 5 个 epochs 保存模型的权重
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True, period=1)
    model = creat_pcg_model(dim, k[dim - 1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    bg, ed = 0, 1280
    for i in range(epochs[dim - 1]):
        while ed < y.shape[0]:
            if dim in [1, 2]:
                model.fit(x=x[bg:ed], y=y[bg:ed], batch_size=batch[dim - 1], epochs=1,
                          shuffle=True, validation_split=0.1, verbose=2, callbacks=[tsbd_callback, ckpt_callback])
            else:
                model.fit(x=(x[0][bg:ed], x[1][bg:ed]), y=y[bg:ed], batch_size=batch[dim - 1], epochs=1,
                          shuffle=True, validation_split=0.1, verbose=2, callbacks=[tsbd_callback, ckpt_callback])
            bg = ed
            ed += 1280
    model.summary()
    # model.save("logs\\model" + str(dim) + "D.h5", save_format='tf')
    return None


def predict_one_sample_step2(probs):
    return int(np.mean(probs > 0.5) > 0.5)


def eval_model(dim=1, x=None, y=None):
    print('Go to evaluate model' + str(dim) + 'D!')
    y = y.astype(np.int)
    predicts = np.zeros_like(y, dtype=np.int)

    model = creat_pcg_model(dim, k[dim - 1])
    latest = tf.train.latest_checkpoint("logs\\model" + str(dim) + "D\\ckpt")
    model.load_weights(latest)
    #### model = tf.keras.models.load_model("logs\\model" + str(dim) + "D.h5")

    for i in range(y.shape[0]):
        xi = x[i] if dim in [1, 2] else (x[0][i], x[1][i])
        prob = model.predict(xi)
        predicts[i] = predict_one_sample_step2(prob)
    y = y[:]
    accuracy = np.mean(predicts == y) * 100
    pos_num = int(np.sum(predicts == 1))
    neg_num = int(np.sum(predicts == 0))
    true_num = int(np.sum(y == 1))
    false_num = int(np.sum(y == 0))
    false_pos = np.sum((y == 0) * (predicts == 1)) / false_num * 100
    false_neg = np.sum((y == 1) * (predicts == 0)) / true_num * 100
    precision = np.sum((y == 1) * (predicts == 1)) / pos_num * 100
    recall = np.sum((y == 1) * (predicts == 1)) / true_num * 100
    F1score = 2 * precision * recall / (precision + recall) / 100
    print(str(dim) + 'D模型：')
    print('总样本%d个，正例%d个，负例%d个。\n识别出阳例%d个，阴例%d个。\n正确率%0.1f%%，假阳率%0.1f%%，假阴率%0.1f%%。' % (
        y.shape[0], true_num, false_num, pos_num, neg_num, accuracy, false_pos, false_neg))
    print('精确度%0.1f%%，召回率%0.1f%%，F1 score: %0.3f' % (precision, recall, F1score))
    with open(str(dim) + 'D模型.txt', 'w') as f1:
        f1.write('总样本%d个，正例%d个，负例%d个。\n识别出阳例%d个，阴例%d个。\n正确率%0.1f%%，假阳率%0.1f%%，假阴率%0.1f%%。' % (
            y.shape[0], true_num, false_num, pos_num, neg_num, accuracy, false_pos, false_neg))
        f1.write('精确度%0.1f%%，召回率%0.1f%%，F1 score: %0.3f' % (precision, recall, F1score))
    plt.plot(y, label='y')
    plt.plot(predicts, label='predicts')
    plt.legend()
    plt.show()
    return None


output_types = [(tf.float32, tf.int64),
                (tf.float32, tf.int64),
                ((tf.float32, tf.float32), tf.int64)]
output_shapes = [((10000, 1), ()),
                 ((128, 128, 1), ()),
                 (((10000, 1), (128, 128, 1)), ())]
epochs = np.array([1, 1, 1]) * 5
batch = [32, 32, 32]
k = np.array([1, 1, 1]) * 1

# train h5py
if __name__ == '__main__':
    with h5py.File('cincset1.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        labels = h5f['label1d']  # labels = h5f['label2d']
        #
        train_model_h5py_generate(dim=1, x=xs, y=labels)
        train_model_h5py_generate(dim=2, x=specs, y=labels)
        train_model_h5py_generate(dim=3, x=(xs, specs), y=labels)

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
        # train_model_small(dim=2, x=specs[:], y=labels[:])
        # train_model_small(dim=3, x=(xs[:], specs[:]), y=labels[:])

# evaluate
if __name__ == '__main__':
    # with open('val.pkl', 'rb') as f:
    #     d = pickle.load(f)
    #     x_train_for_val, spec_train_for_val, y_train_for_val, x_val, spec_val, y_val = \
    #         d['x_train_for_val'], d['spec_train_for_val'], d['y_train_for_val'], \
    #         d['x_val'], d['spec_val'], d['y_val']
    #     eval_model(dim=1, x=x_val, y=y_val)
    #     eval_model(dim=2, x=spec_val, y=y_val)
    #     eval_model(dim=3, x=(x_val, spec_val), y=y_val)

    with h5py.File('val.h5', 'r') as h5f:
        x_val = h5f['x_val']
        spec_val = h5f['spec_val']
        y_val = h5f['y_val']
        x_train_for_val = h5f['x_train_for_val']
        spec_train_for_val = h5f['spec_train_for_val']
        y_train_for_val = h5f['y_train_for_val']
        eval_model(dim=1, x=x_val, y=y_val)
        eval_model(dim=2, x=spec_val, y=y_val)
        eval_model(dim=3, x=(x_val, spec_val), y=y_val)
