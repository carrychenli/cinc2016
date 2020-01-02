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
from common import eval_model_common


def train_eval_data(dim=1, x=None, y=None, x_val=None, y_val=None):
    print('Go to train model' + str(dim) + 'D!')

    model = creat_pcg_model(dim, k[dim - 1])
    criteon = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # training
    # index = np.arange(y.shape[0])
    for epoch in range(epochs[dim - 1]):
        # index = np.random.shuffle(index)
        for batch_id in range(y.shape[0] // batch[dim - 1]):
            bg = batch_id * batch[dim - 1]
            ed = bg + batch[dim - 1]
            # chose = [bg:ed]
            xi = x[bg:ed] if dim in [1, 2] else (x[0][bg:ed], x[1][bg:ed])
            yi = y[bg:ed]
            yii = yi
            yi = tf.squeeze(yi)
            with tf.GradientTape() as tape:
                logits = model(xi)
                loss = criteon(yi, logits)
                metric.update_state(yi, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            # MUST clip gradient here or it will disconverge!
            # grads = [tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if batch_id % 10 == 0:
                print("model%dD. Epoch:%d/%d, batch: %d/%d" % (
                    dim, epoch + 1, epochs[dim - 1], batch_id, y.shape[0] // batch[dim - 1]))
                print(epoch, batch_id, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()
    print('train finish!')

    # eval
    # model_name = "logs_tape\\model" + str(dim) + "D.h5"
    # model.save(model_name)
    # model = tf.keras.models.load_model(model_name)

    model_name = "logs_tape\\model" + str(dim) + "D.h5"
    model.save_weights(model_name)
    model = creat_pcg_model(dim, k[dim - 1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(xi, yii, epochs=1, batch_size=yii.shape[0], verbose=2)
    model.load_weights(model_name)

    eval_model_common(model, dim, x_val, y_val)
    return model


output_types = [(tf.float32, tf.int64),
                (tf.float32, tf.int64),
                ((tf.float32, tf.float32), tf.int64)]
output_shapes = [((10000, 1), ()),
                 ((128, 128, 1), ()),
                 (((10000, 1), (128, 128, 1)), ())]
epochs = np.array([1, 1, 1]) * 3
batch = [32, 32, 32]
k = np.array([1, 1, 1]) * 8

# train h5py
if __name__ == '__main__':
    with h5py.File('cincsetx.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        labels = h5f['label1d']  # labels = h5f['label2d']
        with open('val.pkl', 'rb') as f:
            d = pickle.load(f)
            # x_val, spec_val, y_val  = \  # for train data
            #     d['x_train_for_val'], d['spec_train_for_val'], d['y_train_for_val']
            x_val, spec_val, y_val = d['x_val'], d['spec_val'], d['y_val']  #
            train_eval_data(dim=1, x=xs, y=labels, x_val=x_val, y_val=y_val)
            train_eval_data(dim=2, x=specs, y=labels, x_val=spec_val, y_val=y_val)
            train_eval_data(dim=3, x=(xs, specs), y=labels, x_val=(x_val, spec_val), y_val=y_val)
