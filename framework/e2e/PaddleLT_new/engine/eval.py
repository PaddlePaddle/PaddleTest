#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import numpy as np
import paddle
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class LayerEval(object):
    """
    构建Layer评估的通用类
    """

    # def __init__(self, testing, case, layer):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.testing = testing

        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.net = BuildLayer(layerfile=layerfile).get_layer()

        self.data = BuildData(layerfile=layerfile).get_single_data()

    def dy_eval(self):
        """dygraph eval"""
        reset(self.seed)

        # self.net.eval()
        logit = self.net(*self.data)
        return logit

    def dy2st_eval(self):
        """dy2st eval"""
        reset(self.seed)

        net = paddle.jit.to_static(self.net, full_graph=True)
        # net.eval()
        logit = net(*self.data)
        return logit

    def dy2st_eval_cinn(self):
        """dy2st eval"""
        reset(self.seed)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        net = paddle.jit.to_static(self.net, build_strategy=build_strategy, full_graph=True)
        # net.eval()
        logit = net(*self.data)
        return logit
