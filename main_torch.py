#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: main_torch.py
# @Time    : 2019/12/31 15:59
# @Description:
"""

"""

from ndtpy.tools import list_all_files, get_fig_title, maximize_figure,func_timer

from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False

from pcg_model_torch import create_pcg_model
import h5py, pickle
import torch.optim as optim
from torch import nn
import torch


def train_model(dim, x, y):
    print('Train model' + str(dim) + 'D!')
    net = create_pcg_model(dim=dim, k=k)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # print('hereh1')
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    nums = y.shape[0] // batch_size
    for epoch in range(epoches):
        print('model%dD, epoch: %d/%d' % (dim, epoch + 1, epoches))
        running_loss = 0.0
        for i in range(nums):
            optimizer.zero_grad()
            # forward + backward + optimize
            if dim in [1, 2]:
                xi = torch.tensor(x[i * batch_size:(i + 1) * batch_size]).to(device)
                # print(xi.shape)
            else:
                xi = (torch.tensor(x[0][i * batch_size: (i + 1) * batch_size]).to(device),
                      torch.tensor(x[1][i * batch_size: (i + 1) * batch_size]).to(device))
            outputs = net(xi)
            yi = torch.tensor(y[i * batch_size:(i + 1) * batch_size]).to(device)
            loss = criterion(outputs, yi)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    print('Finished Training!')
    PATH = 'logs_torch/torch_model' + str(dim) + 'D.pth'
    torch.save(net.state_dict(), PATH)
    return None


def predict_one_sample_step2(probs):
    return int(np.mean(probs > 0.5) > 0.5)


def eval_model(dim, x, y):
    net = create_pcg_model(dim, k=k)
    PATH = 'logs_torch/torch_model' + str(dim) + 'D.pth'
    net.load_state_dict(torch.load(PATH))
    predicts = np.zeros_like(y)
    with torch.no_grad():
        for i in range(y.shape[0]):
            xi = torch.tensor(x[i], dtype=torch.float32) if dim in [1, 2] else \
                (torch.tensor(x[0][i], dtype=torch.float32), torch.tensor(x[1][i], dtype=torch.float32))
            prob = net(xi).numpy()
            # print(prob)
            # exit()
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


epoches = 5
batch_size = 32
k = 8
if __name__ == '__main__':
    with h5py.File('torch_cincsetx.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        labels = h5f['label1d']  # labels = h5f['label2d']
        #
        train_model(dim=1, x=xs, y=labels)
        train_model(dim=2, x=specs, y=labels)
        train_model(dim=3, x=(xs, specs), y=labels)

# evaluate
if __name__ == '__main__':
    with open('val_torch.pkl', 'rb') as f:
        d = pickle.load(f)
        # x, spec, y = \  # for train data
        #     d['x_train_for_val'], d['spec_train_for_val'], d['y_train_for_val']
        x, spec, y = d['x_val'], d['spec_val'], d['y_val']
        eval_model(dim=1, x=x, y=y)
        eval_model(dim=2, x=spec, y=y)
        eval_model(dim=3, x=(x, spec), y=y)
