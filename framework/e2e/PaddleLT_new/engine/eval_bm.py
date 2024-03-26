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
from engine.xtools import reset, trimmean, mean, best, best_top_k
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class LayerEvalBM(object):
    """
    构建Layer评估的性能通用类
    """

    # def __init__(self, testing, case, layer):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.set_device(str(self.device))
        self.device_id = os.environ.get("PLT_DEVICE_ID")
        self.perf_repeat = int(os.environ.get("PLT_BM_REPEAT", "10"))
        self.perf_statis = os.environ.get("PLT_BM_STATIS", "trimmean")

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

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
        timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=1)
            total_time_list.append(total_time)

        time_res = eval(self.perf_statis)(data_list=total_time_list)
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
        timeit.timeit(lambda: _perf(self.data), number=int(self.perf_repeat * 0.2))
        for i in range(self.perf_repeat):
            total_time = timeit.timeit(lambda: _perf(self.data), number=1)
            total_time_list.append(total_time)

        time_res = eval(self.perf_statis)(data_list=total_time_list)
        return time_res
