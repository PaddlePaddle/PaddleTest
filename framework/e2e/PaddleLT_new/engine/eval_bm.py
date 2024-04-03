#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import os
import timeit
import numpy as np
import paddle
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData
from tools.res_save import save_pickle
from tools.statistics import trimmean, mean, best, best_top_k, perf_by_step


class LayerEvalBM(object):
    """
    构建Layer评估的性能通用类
    """

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.set_device(str(self.device))
        # paddle.set_device("{}:{}".format(str(self.device), str(device_id)))

        self.perf_repeat = int(os.environ.get("PLT_BM_REPEAT", "100"))
        self.perf_statis = os.environ.get("PLT_BM_STATIS", "trimmean")
        self.timeit_num = int(os.environ.get("TIMEIT_NUM", "1"))
        self.statis_times = 100
        self.statis_round = 6

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile
        self.net = BuildLayer(layerfile=layerfile).get_layer()

        self.data = BuildData(layerfile=layerfile).get_single_tensor()

    def dy_eval_perf(self):
        """dygraph eval"""
        reset(self.seed)

        # self.net.eval()
        net = self.net

        def _perf(input_data):
            logit = net(*input_data)
            return logit

        total_time_list = []
        # 预热
        timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * self.timeit_num * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=self.timeit_num)
            total_time_list.append(total_time)

        save_pickle(data_list=total_time_list, filename="dy_eval_perf_" + self.layerfile)
        time_res = eval(self.perf_statis)(data_list=total_time_list)
        time_res = round(time_res * self.statis_times, self.statis_round)
        return time_res

    def dy2st_eval_cinn_perf(self):
        """dy2st eval"""
        reset(self.seed)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        net = paddle.jit.to_static(self.net, build_strategy=build_strategy, full_graph=True)

        # net.eval()
        def _perf(input_data):
            logit = net(*input_data)
            return logit

        total_time_list = []
        # 预热
        timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * self.timeit_num * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=self.timeit_num)
            total_time_list.append(total_time)

        save_pickle(data=total_time_list, filename="dy2st_eval_cinn_perf_" + self.layerfile + "_total_time_list")
        # 画图
        perf_by_step(
            data_list=total_time_list,
            step_scale=[0.1, 0.5, 1],
            filename="dy2st_eval_cinn_perf_" + self.layerfile + "_by_step",
        )

        time_res = eval(self.perf_statis)(data_list=total_time_list)
        time_res = round(time_res * self.statis_times, self.statis_round)
        return time_res
