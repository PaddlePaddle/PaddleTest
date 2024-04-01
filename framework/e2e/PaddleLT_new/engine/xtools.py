#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
常用tools
"""
import numpy as np
import paddle


def reset(seed):
    """
    重置模型图
    :param seed: 随机种子
    :return:
    """
    paddle.enable_static()
    paddle.disable_static()
    paddle.seed(seed)
    np.random.seed(seed)
    np.set_printoptions(threshold=5, edgeitems=3)


# # 多种统计学计算策略
# def trimmean(data_list, ratio=0.2):
#     """
#     掐头去尾求平均
#     :param data_list: 输入的data list, 多次试验的结果集合
#     """
#     head = int(len(data_list) * ratio)
#     tail = int(len(data_list) - len(data_list) * ratio)
#     res = sum(sorted(data_list)[head:tail]) / (tail - head)
#     return res


# def mean(data_list):
#     """
#     求平均值
#     :param data_list:
#     :return:
#     """
#     res = sum(data_list) / len(data_list)
#     return res


# def best(data_list):
#     """
#     找出耗时最少的一次试验结果
#     :param data_list: 输入的data list, 多次试验的结果集合
#     :return: 最少的时间
#     """
#     # print('data_list is: ', data_list)
#     res = min(data_list)
#     return res


# def best_top_k(data_list, ratio=0.2):
#     """
#     求最优top k的平均值，默认ratio=0.2
#     :param data_list: 输入的data list, 多次试验的结果集合
#     """
#     head = int(len(data_list) * ratio)
#     res = sum(sorted(data_list)[:head]) / head
#     return res

# # list 保存/加载 为pickle
# def save_pickle(data_list, filename):
#     """
#     """
#     with open(filename, 'wb') as f:
#         # 使用pickle的dump函数将列表写入文件
#         pickle.dump(data_list, f)

# def load_pickle(filename):
#     """
#     """
#     with open(filename, 'rb') as f:
#         # 使用pickle的load函数从文件中加载列表
#         loaded_data = pickle.load(f)

#     return loaded_data
