#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: common.py
# @Time    : 2019/12/25 17:48
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

def eval_model_common(model, dim, x, y):
    predicts = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        xi = x[i] if dim in [1, 2] else (x[0][i], x[1][i])
        prob = model.predict(xi)
        predicts[i] = predict_one_sample_step2(prob)
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


def predict_one_sample_step2(probs):
    return int(np.mean(probs > 0.5) > 0.5)


def generate_dataset(x, y, mode='train'):
    if mode == 'train':
        i = 0
        ed = int(y.shape[0] * 0.9)
    else:
        i = int(y.shape[0] * 0.9)
        ed = y.shape[0]
    while i < ed:
        data = (x[i], y[i])
        i += 1
        yield data


def generate_dataset_3D(x0, x1, y, mode='train'):
    if mode == 'train':
        i = 0
        ed = int(y.shape[0] * 0.9)
    else:
        i = int(y.shape[0] * 0.9)
        ed = y.shape[0]
    while i < ed:
        data = ((x0[i], x1[i]), y[i])
        i += 1
        yield data