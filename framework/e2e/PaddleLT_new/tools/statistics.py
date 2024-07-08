#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
常用统计学计算策略
"""

import numpy as np
import matplotlib.pyplot as plt

# 多种统计学计算策略
def trimmean(data_list, ratio=0.2):
    """
    掐头去尾求平均
    :param data_list: 输入的data list, 多次试验的结果集合
    """
    head = int(len(data_list) * ratio)
    tail = int(len(data_list) - len(data_list) * ratio)
    res = sum(sorted(data_list)[head:tail]) / (tail - head)
    return res


def mean(data_list):
    """
    求平均值
    :param data_list:
    :return:
    """
    res = sum(data_list) / len(data_list)
    return res


def best(data_list):
    """
    找出耗时最少的一次试验结果
    :param data_list: 输入的data list, 多次试验的结果集合
    :return: 最少的时间
    """
    # print('data_list is: ', data_list)
    res = min(data_list)
    return res


def best_top_k(data_list, ratio=0.2):
    """
    求最优top k的平均值，默认ratio=0.2
    :param data_list: 输入的data list, 多次试验的结果集合
    """
    head = int(len(data_list) * ratio)
    res = sum(sorted(data_list)[:head]) / head
    return res


# list等分
def split_list(lst, n):
    """
    将列表按顺序划分为 n 份
    Args:
        lst (list): 待划分的列表
        n (int): 划分的份数
    Returns:
        res (list): 划分后的列表，其中每个元素为原列表的 1/n 部分
    """
    if not isinstance(lst, list) or not isinstance(n, int) or len(lst) == 0 or n <= 0:
        return []
    quotient, remainder = divmod(len(lst), n)
    res = [[] for _ in range(n)]
    for i, value in enumerate(lst):
        index = i % n
        res[index].append(value)
    return res


# 作图
def perf_by_step(data_list, step_scale, filename):
    """
    横坐标为执行轮数, 纵坐标为耗时
    data_list: 耗时list, 以轮数为index
    step_scale: 以轮数的一定比例作图, list
    filename: 图片文件名
    """

    for percent_scale in step_scale:
        length = len(data_list)
        percent_count = int(length * percent_scale)
        # 提取横坐标（索引）和纵坐标（元素值）
        x_values = range(percent_count)
        y_values = data_list[:percent_count]

        # 绘制图形
        plt.clf()
        plt.plot(x_values, y_values, marker="o")  # 使用圆点作为数据点的标记
        plt.xlabel("Step-{}".format(percent_count))  # 设置横坐标轴标签
        plt.ylabel("Time-Consuming")  # 设置纵坐标轴标签
        plt.title(filename)  # 设置图形标题
        plt.grid(True)  # 显示网格线
        plt.savefig("{}_step{}.png".format(filename, percent_count))
        plt.clf()


# 计算四分位范围
def Q1_Q4_range(data_list):
    """
    取四分位数
    """
    # 按升序排列
    sorted_data = sorted(data_list)

    # 计算第一四分位数 Q1
    Q1 = np.percentile(sorted_data, 25)

    # 计算第三四分位数 Q3
    Q3 = np.percentile(sorted_data, 75)

    # 计算四分位数范围 IQR
    IQR = Q3 - Q1

    # 确定取值范围
    k = 1.5  # 可根据需要调整
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return lower_bound, upper_bound


def gsb_rule(res):
    """
    评分标准
    :param res: compare_res对比
    :return:
    """
    pass

    # grade = ""
    # if isinstance(res, str):
    #     grade = res
    # else:
    #     if res <= -1.3:
    #         grade = "worse"
    #     elif -1.3 < res <= -1.15:
    #         grade = "doubt"
    #     elif -1.15 < res <= 1.15:
    #         grade = "equal"
    #     elif res > 1.15:
    #         grade = "better"
    # return grade


def ci_level_reveal(compare_res):
    """
    等级分类
    :param compare_res:
    :return:
    """
    pass
    # grade_dict = {}
    # grade_dict["worse"] = []
    # grade_dict["doubt"] = []
    # grade_dict["equal"] = []
    # grade_dict["better"] = []

    # for case_name, compare_dict in compare_res.items():
    #     tmp = {}
    #     # grade = gsb_rule(res=compare_dict["forward"])
    #     # tmp[compare_dict["latest_api"]] = compare_dict["forward"]
    #     grade = gsb_rule(res=compare_dict["best_total"])
    #     tmp[compare_dict["latest_api"]] = compare_dict["best_total"]
    #     grade_dict[grade].append(tmp)

    # return grade_dict
