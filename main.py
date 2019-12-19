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
from pcg_model import PCG, PCGONLY
import tensorflow as tf

register_matplotlib_converters()
plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False

import h5py

if __name__ == '__main__':
    with h5py.File('cincset.h5', 'r') as h5f:
        xs = h5f['x']
        specs = h5f['spectrogram']
        # labels = h5f['label2d']
        labels = h5f['label1d']

        print('go1, model1D')
        model1D = PCGONLY(dim=1)
        model1D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model1D.fit(x=xs[:], y=labels[:], batch_size=64, epochs=1)
        y1 = model1D.predict(x=xs[0:2])

        print('go2, model2D')
        model2D = PCGONLY(dim=2)
        model2D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model2D.fit(x=specs[:], y=labels[:], batch_size=64, epochs=1)
        y2 = model2D.predict(x=specs[0:2])

        print('go3, model')
        model = PCG()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x=(xs[:], specs[:]), y=labels[:], batch_size=64, epochs=1)
        y3 = model.predict(x=(xs[0:2], specs[0:2]))
