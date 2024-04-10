#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import os
import numpy as np
import torch
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class TorchLayerEval(object):
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
        torch.device(str(self.device))

        self.testing = testing
        self.model_dtype = self.testing.get("model_dtype")

    #     paddle.set_default_dtype(self.model_dtype)

    #     self.layerfile = layerfile
    #     self.data = BuildData(layerfile=self.layerfile).get_single_data()

    # def _net_instant(self):
    #     """get net and data"""
    #     reset(self.seed)
    #     net = BuildLayer(layerfile=self.layerfile).get_layer()
    #     return net

    # def dy_eval(self):
    #     """dygraph eval"""
    #     net = self._net_instant()
    #     # net.eval()
    #     logit = net(*self.data)
    #     return logit

    # def dy2st_eval(self):
    #     """dy2st eval"""
    #     net = self._net_instant()
    #     st_net = paddle.jit.to_static(net, full_graph=True)
    #     # net.eval()
    #     logit = st_net(*self.data)
    #     return logit

    # def dy2st_eval_cinn(self):
    #     """dy2st eval"""
    #     net = self._net_instant()

    #     build_strategy = paddle.static.BuildStrategy()
    #     build_strategy.build_cinn_pass = True
    #     cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
    #     # net.eval()
    #     logit = cinn_net(*self.data)
    #     return logit
