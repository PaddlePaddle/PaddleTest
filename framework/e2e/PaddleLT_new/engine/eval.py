#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import os
import numpy as np
import paddle
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class LayerEval(object):
    """
    构建Layer评估的通用类
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

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile

    def _net_data(self):
        """get net and data"""
        reset(self.seed)
        net = BuildLayer(layerfile=self.layerfile).get_layer()
        data = BuildData(layerfile=self.layerfile).get_single_data()
        return net, data

    def dy_eval(self):
        """dygraph eval"""
        net, data = self._net_data()
        # net.eval()
        logit = net(*data)
        return logit

    def dy2st_eval(self):
        """dy2st eval"""
        net, data = self._net_data()
        st_net = paddle.jit.to_static(net, full_graph=True)
        # net.eval()
        logit = st_net(*data)
        return logit

    def dy2st_eval_cinn(self):
        """dy2st eval"""
        net, data = self._net_data()

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
        # net.eval()
        logit = cinn_net(*data)
        return logit
