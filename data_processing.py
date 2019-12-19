#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: data_processing.py
# @Time    : 2019/12/16 17:30
# @Description:
"""

"""

from ndtpy.tools import list_all_files, get_fig_title, maximize_figure

from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wave
from sklearn import preprocessing


def read_wav(wav_file):
    wf = wave.open(wav_file, 'rb')
    str_data = wf.readframes(wf.getnframes())
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wf.close()
    return wave_data


def read_one_folder(folder='validation'):
    files = list_all_files(folder, '.wav')
    data, frames = [], []
    for fid, wav_file in enumerate(files):
        wave_data = read_wav(wav_file)
        data.append(wave_data)
    label = pd.read_csv(folder + '\\REFERENCE.csv', header=None)[1].to_numpy()
    return data, label


def read_train():
    folder = 'training\\training-'
    datas, labels = [], np.empty(shape=[0, ])
    subs = ['a', 'b', 'c', 'd', 'e', 'f']
    for sub in subs:
        data, label = read_one_folder(folder + sub)
        datas.extend(data)
        labels = np.r_[labels, label]
    return datas, labels


def sub_data_cut(data, label, sliding_rate=100):
    assert 1 <= sliding_rate <= window
    sliding = int(window // sliding_rate)
    length = (data.shape[0] - window) // sliding + 1
    x = np.zeros(shape=(length, window))
    y = np.zeros(shape=(length,)) + label
    for i in range(length):
        bg = i * sliding
        ed = bg + window
        x[i] = data[bg:ed]
    return x, y


def prepare_train_data():
    datas, labels = read_train()
    # pos_neg = sum(labels > 0) / sum(labels < 0)
    x_train, y_train = np.empty(shape=(0, window)), np.empty(shape=(0,))
    for data, label in zip(datas, labels):
        if label == 1:
            x, y = sub_data_cut(data, label, sliding_rate=pos_sliding_rate)
        elif label == -1:
            x, y = sub_data_cut(data, label, sliding_rate=pos_sliding_rate * pos_neg)
        else:
            print('error1: impossible!')
            exit(1)
        x_train = np.r_[x_train, x]
        y_train = np.r_[y_train, y]
    return x_train, y_train


def validation_data_cut(data, label):
    x, _ = sub_data_cut(data, label, sliding_rate=1)
    x = np.append(x, data[-window:])
    return x


def prepare_validation_data():
    datas, labels = read_one_folder(folder='validation')
    x_val = []
    for data, label in zip(datas, labels):
        x = validation_data_cut(data, label)
        x_val.append(x)
    return x_val, labels


def prepare_train_data_for_val():
    datas, labels = read_train()
    x_train_for_val = []
    for data, label in zip(datas, labels):
        x = validation_data_cut(data, label)
        x_train_for_val.append(x)
    return x_train_for_val, labels


import pickle

framerate = 2000
window = framerate * 5
pos_neg = 1 / 4
pos_sliding_rate = 1000

if __name__ == '__main__':
    x_train, y_train = prepare_train_data()
    d1 = {'x_train': x_train, 'y_train': y_train}
    with open('val.pkl', 'wb') as f1:
        pickle.dump(d1, f1)

    # x_val, y_val = prepare_validation_data()
    # x_train_for_val, y_train_for_val = prepare_train_data_for_val()
    # d2 = {'x_train_for_val': x_train_for_val,
    #       'y_train_for_val': y_train_for_val,
    #       'x_val': x_val, 'y_val': y_val}
    # with open('val.pkl', 'wb') as f2:
    #     pickle.dump(d2, f2)

    # with open('val.pkl', 'rb') as f2:
    #     d2 = pickle.load(f2)
    #     x_train_for_val, y_train_for_val, x_val, y_val = \
    #         d2['x_train_for_val'], d2['y_train_for_val'], d2['x_val'], d2['y_val']
