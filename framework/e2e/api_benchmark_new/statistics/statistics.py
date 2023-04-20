#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
trimmean 掐头去尾求平均
"""


class Statistics(object):
    """
    多种统计学计算策略
    """

    def __init__(self):
        """
        初始化
        """
        # self.data_list = data_list
        self.ACCURACY = "%.6g"

    def trimmean(self, data_list, ratio=0.2):
        """
        掐头去尾求平均
        :param data_list: 输入的data list, 多次试验的结果集合
        """
        head = int(len(data_list) * ratio)
        tail = int(len(data_list) - len(data_list) * ratio)
        res = sum(sorted(data_list)[head:tail]) / (tail - head)
        return res

    def best(self, data_list):
        """
        找出耗时最少的一次试验结果
        :param data_list: 输入的data list, 多次试验的结果集合
        :return: 最少的时间
        """
        # print('data_list is: ', data_list)
        res = min(data_list)
        return res
