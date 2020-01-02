#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @Project : cinc2016 
# @FileName: forleetcode.py
# @Time    : 2019/12/30 9:55
# @Description:
"""

"""

import numpy as np
from copy import deepcopy


# class Solution:
#     def exist_state(self, board, state, word):
#         if len(word) == 1:
#
#     def exist(self, board: list[list[str]], word: str) -> bool:
#         row = len(board)
#         if row == 0 or len(word) == 0:
#             return False
#         col = len(board[0])
#         if col == 0:
#             return False
#         state = np.ones(shape=(row, col), dtype=np.bool)

def removeduplicates(nums):
    if len(nums) <= 2:
        return len(nums)
    cur_num = nums[0]
    cur_time = 1
    length = 1
    for num in nums[1:]:
        if cur_num == num:
            cur_time += 1
            if cur_time <= 2:
                length += 1
        else:
            cur_num = num
            cur_time = 1
            length += 1
    return length

issubclass()