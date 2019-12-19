#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: data_preprocessing_v2.py
# @Time    : 2019/12/17 16:45
# @Description:
"""

"""

from ndtpy.tools import list_all_files, get_fig_title, maximize_figure, func_timer

from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import wave
from sklearn import preprocessing
import h5py
import pickle


def read_wav(wav_file):
    wf = wave.open(wav_file, 'rb')
    str_data = wf.readframes(wf.getnframes())
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wf.close()
    return wave_data


def read_one_folder(folder='validation'):
    files = list_all_files(folder, '.wav')
    label = pd.read_csv(folder + '\\REFERENCE.csv', header=None)[1].to_numpy()
    frame = np.zeros_like(label)
    data = []
    for fid, wav_file in enumerate(files):
        wave_data = read_wav(wav_file)
        data.append(wave_data)
        frame[fid] = wave_data.shape[0] - window + 1
    return data, label, frame


def read_train():
    folder = 'training\\training-'
    datas, labels, frames = [], np.empty(shape=[0, ]), np.empty(shape=[0, ])
    subs = ['a', 'b', 'c', 'd', 'e', 'f']
    for sub in subs:
        data, label, frame = read_one_folder(folder + sub)
        datas.extend(data)
        labels = np.r_[labels, label]
        frames = np.r_[frames, frame]
    return datas, labels, frames


def random_one_data_cut(data, frame):
    bg = np.random.randint(0, frame)
    x = data[bg:bg + window]
    x = x - x.mean()
    x = x / (np.abs(x).max())
    return x


def get_random_pos(frame_cumsum):
    rd = np.random.rand() * frame_cumsum[-1]
    length = frame_cumsum.shape[0]
    for pos in range(length):
        if rd <= frame_cumsum[pos]:
            return pos
    return 0


def get_cumsum(frames, index):
    """
    带sqrt的惩罚，不至于数据太大的样本被采集太多次
    """
    tmp = frames[index]
    tmp = (tmp.min() * np.sqrt(tmp / tmp.min())).astype(np.int)
    return np.cumsum(tmp)


def get_spectrogram(xs):
    local_window, local_sliding = 200, 77  # 配齐 128 * 128
    length = (window - local_window) // local_sliding + 1
    # print(length)
    spectrogram = np.zeros(shape=(length, length))
    for i in range(length):
        bg = i * local_sliding
        x = xs[bg:bg + local_window]
        spectrogram[i] = np.abs(np.fft.fft(x - x.mean(), n=nfft))[1:nfft // 2 + 1]
    spectrogram /= np.max(np.max(spectrogram))
    return spectrogram


@func_timer
def prepare_train_data():
    datas, labels, frames = read_train()
    pos_index, neg_index = np.where(labels > 0)[0], np.where(labels < 0)[0]
    pos_frame_cumsum = get_cumsum(frames, pos_index)
    neg_frame_cumsum = get_cumsum(frames, neg_index)
    try:
        h5f = h5py.File(h5fn, 'w-')
        x_set = h5f.create_dataset("x", shape=(target_num, window, 1),
                                   maxshape=(None, window, 1), chunks=(128, window, 1))
        spec_set = h5f.create_dataset("spectrogram", shape=(target_num, nfft // 2, nfft // 2, 1),
                                      maxshape=(None, nfft // 2, nfft // 2, 1),
                                      chunks=(128, nfft // 2, nfft // 2, 1))
        label_set1d = h5f.create_dataset("label1d", shape=(target_num,), maxshape=(None,), chunks=(128,))
        label_set2d = h5f.create_dataset("label2d", shape=(target_num, 2), maxshape=(None, 2), chunks=(128, 2))
        i0 = 0
    except:
        h5f = h5py.File(h5fn, 'a')
        x_set = h5f['x']
        spec_set = h5f['spectrogram']
        label_set1d = h5f['label1d']
        label_set2d = h5f['label2d']
        i0 = label_set1d.shape[0]
        x_set.resize([target_num + i0, window, 1])
        spec_set.resize([target_num + i0, nfft // 2, nfft // 2, 1])
        label_set1d.resize([target_num + i0, ])
        label_set2d.resize([target_num + i0, 2])

    for i in range(target_num):
        if np.random.rand() <= 0.5:  # 抽取一个pos
            pos = get_random_pos(pos_frame_cumsum)
            pos = pos_index[pos]
            label2d = np.array([1, 0])
            label1d = 1
        else:  # 抽取一个 neg
            pos = get_random_pos(neg_frame_cumsum)
            pos = neg_index[pos]
            label2d = np.array([0, 1])
            label1d = 0
        data, frame = datas[pos], frames[pos]
        x = random_one_data_cut(data, frame)

        spectrogram = get_spectrogram(x)
        x_set[i0 + i] = x[:, np.newaxis]
        spec_set[i0 + i] = spectrogram[:, :, np.newaxis]
        label_set1d[i + i0] = label1d
        label_set2d[i + i0] = label2d
        if i % int(target_num * 0.01) == 0:
            print('%d/%d: %d%%' % (i, int(target_num), int(i * 100 / target_num)))
    print('Finish!\n Total samples quantity is %d, the new added is %d.' % (label_set1d.shape[0], target_num))
    h5f.close()


"""
以下函数只于测试集的预处理有关
"""


def validation_data_cut(data):
    length = data.shape[0] // window + 1
    x = np.zeros(shape=(length, window))
    for i in range(length - 1):
        bg = i * window
        ed = bg + window
        x[i] = data[bg:ed]
    x[-1] = data[-window:]
    return x


def prepare_validation_data(func):
    datas, labels, _ = func()
    x_val, spec_val = [], []
    for data in datas:
        x = validation_data_cut(data)
        spec = get_spectrogram(x[0])
        specs = np.zeros(shape=(x.shape[0], spec.shape[0], spec.shape[1]))
        specs[0] = spec
        for i in range(x.shape[0] - 1):
            specs[i + 1] = get_spectrogram(x[i + 1])
        x_val.append(x)
        spec_val.append(specs)
    return x_val, spec_val, labels


framerate = 2000
window = framerate * 5
target_num = int(8e5)
nfft = 256
h5fn = 'cincset.h5'

if __name__ == '__main__':
    prepare_train_data()

    """"
    以下 for val
    """
    # h5py write
    if False:
        x_val, spec_val, y_val = prepare_validation_data(read_one_folder)
        x_train_for_val, spec_train_for_val, y_train_for_val = prepare_validation_data(read_train)
        lenval = len(y_val)
        lentrain = len(y_train_for_val)
        with h5py.File('val.h5', 'w') as h5f:
            x_val_set = h5f.create_dataset("x_val", shape=(lenval,))
            spec_val_set = h5f.create_dataset("spec_val", shape=(lenval,))
            y_val_set = h5f.create_dataset("y_val", shape=(lenval,))
            x_train_for_val_set = h5f.create_dataset("x_train_for_val", shape=(lentrain,))
            spec_train_for_val_set = h5f.create_dataset("spec_train_for_val", shape=(lentrain,))
            y_train_for_val_set = h5f.create_dataset("y_train_for_val", shape=(lentrain,))
            x_val_set[:] = x_val
            spec_val_set[:] = spec_val
            y_val_set[:] = y_val
            x_train_for_val_set[:] = x_train_for_val
            spec_train_for_val_set[:] = spec_train_for_val
            y_train_for_val_set[:] = y_train_for_val

    # h5py read
    if False:
        with h5py.File('val.h5', 'r') as h5f:
            x_val = h5f['x_val']
            spec_val = h5f['spec_val']
            y_val = h5f['y_val']
            x_train_for_val = h5f['x_train_for_val']
            spec_train_for_val = h5f['spec_train_for_val']
            y_train_for_val = h5f['y_train_for_val']

    # pickle write
    if False:
        d = {'x_train_for_val': x_train_for_val,
             'spec_train_for_val': spec_train_for_val,
             'y_train_for_val': y_train_for_val,
             'x_val': x_val, 'spec_val': spec_val, 'y_val': y_val}
        with open('val.pkl', 'wb') as f:
            pickle.dump(d, f)

    # pickle read
    if False:
        with open('val.pkl', 'rb') as f:
            d = pickle.load(f)
            x_train_for_val, spec_train_for_val, y_train_for_val, x_val, spec_val, y_val = \
                d['x_train_for_val'], d['spec_train_for_val'], d['y_train_for_val'], \
                d['x_val'], d['spec_val'], d['y_val']
