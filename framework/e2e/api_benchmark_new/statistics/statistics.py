#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
trimmean 掐头去尾求平均
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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

    def mean(self, data_list):
        """

        :param data_list:
        :return:
        """
        res = sum(data_list) / len(data_list)
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

    def probability_plot(self, data_list):
        """

        :return:
        """
        noise_list = list(map(lambda x: x[0] - self.mean(data_list), zip(data_list)))
        # 计算概率密度函数
        density = gaussian_kde(sorted(noise_list))

        # 生成一组用于绘制概率分布图的值
        xs = np.linspace(min(noise_list), max(noise_list), 200)

        # 绘制概率分布图
        plt.plot(xs, density(xs))

        # 设置横轴、纵轴标签和图表标题
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("Probability Density Plot")

        # 显示图表
        plt.show()
        # 运行这段代码将会生成一个概率分布图，横轴为值，纵轴为值的概率密度。你可以将data变量替换为自己的List，以绘制相应的概率分布图。

    def fish_stat(self, data_list, Nbins=1000):
        """

        :param data_list:
        :return:
        """

        a = data_list
        # X = (np.random.randn(1000)) * 2 + 100
        X = np.array(a)
        # X = np.array(X)

        # Data_std = (np.random.randn(1000))
        Data_std = (X - np.mean(X)) / np.std(X)

        # Nbins = 20
        # %%
        Data_freq = np.zeros(Nbins)
        d_range = int(np.max(np.abs(Data_std))) + 1

        amp = float(Nbins) / (d_range * 2)

        for d in Data_std:
            id_freq = int((d + d_range) * amp)
            Data_freq[id_freq] += 1

        Data_freq = Data_freq / len(data_list)
        # %%

        t_std = np.zeros(Nbins)
        N_std = np.zeros(Nbins)

        for i in range(Nbins):
            t_std[i] = (i - Nbins / 2) / Nbins * np.max(Data_std)
            N_std[i] = np.exp(-t_std[i] * t_std[i]) / np.sqrt(2 * np.pi)

        # %%
        import matplotlib.pyplot as plt

        # plt.pyplot.hist(Data_std, bins= Nbins)
        plt.plot(t_std, N_std)
        plt.plot(t_std, Data_freq)

        plt.show()

    def littlefish(self, data_list, Nbins=50):
        """

        :param data_list:
        :return:
        """
        print("start...")
        # data_list = sorted(data_list)[2000:8000]
        # X = (np.random.randn(10000)) * 2 + 100
        X = np.array(data_list).astype("float64")
        X = np.sort(X)[:-2000]

        Data_std = (X - np.mean(X)) / np.std(X)

        # Data_std = (np.random.randn(1000))  # standardize
        # Nbins = 100  # Number of bins
        ddd = np.sort(Data_std)
        print(ddd[0:10])
        print(ddd[-10:])
        print("np.max(X) is: ", np.max(X))

        # %% culculate pdf of data
        Data_freq = np.zeros(Nbins)
        d_range = int(np.max(np.abs(Data_std))) + 1  # adjust: 0 < d <  max(abs)+1
        # print('ddd[0] is: ', ddd[0])
        # print('ddd[-1] is: ', ddd[-1])

        # if -ddd[0]  > ddd[-1]:
        #     d_range = int(-ddd[0]) + 1
        # else:
        #     d_range = int(ddd[-1]) + 1
        # print('max :   ', np.max(Data_std))
        print("np.max(np.abs(Data_std)) is: ", np.max(np.abs(Data_std)))
        print("d_range is: ", d_range)

        for d in Data_std:  # N_bins freq num
            id_freq = int((d + d_range) * (Nbins) / (d_range * 2))  # 0 <= id < Nbins
            Data_freq[id_freq] += 1

        print("Data_freq is: ", Data_freq[-10:])
        Data_freq = Data_freq / len(Data_std)  # to frequancy
        print("Data_freq is: ", Data_freq)
        # print(np.sum(Data_freq)) # sum should be 1
        # %% plot N(0,1)
        t_std = np.zeros(Nbins)
        N_std = np.zeros(Nbins)

        for i in range(Nbins):
            t_std[i] = (i - Nbins / 2) * d_range * 2 / Nbins  # np.max(Data_std)
            N_std[i] = np.exp(-t_std[i] ** 2 * 0.5) / np.sqrt(2 * np.pi)

        N_std = N_std / np.sum(N_std)

        print(t_std[0:5])
        # %% calculate mean square error

        err = 0  # mean square error
        for i in range(Nbins):
            err += (Data_freq[i] - N_std[i]) ** 2
        err = err / Nbins
        print("mean square error: {}".format(err))
        # %% plot

        plt.plot(t_std, N_std, label="syandard normal N(0,1)")
        plt.plot(t_std, Data_freq, label="pdf of data")
        plt.title(" pdf comparison, mean square error = {}".format(err))
        plt.legend()
        plt.show()
