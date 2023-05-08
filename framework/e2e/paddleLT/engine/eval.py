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

    def __init__(self, testing, case, layer):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.testing = testing
        self.layer = layer
        self.case = case

        self.layer_name = self.layer.get("Layer").get("layer_name")
        self.layer_param = self.layer.get("Layer").get("params")
        self.net = BuildLayer(layer_name=self.layer_name, layer_param=self.layer_param)

        self.data_info = self.layer.get("DataGenerator")
        self.data = BuildData(data_info=self.data_info).get_single_data()

        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

    def dy_eval(self):
        """dygraph eval"""
        reset(self.seed)

        # data_dict = self.input_data[0]  # self.input_data.__getitem__(0)
        data_dict = self.data[0]
        net = self.net.get_layer()
        net.eval()
        logit = net(**data_dict)
        # logit = self.loss_info.get_loss(logit)
        return logit

    def dy2st_eval(self):
        """dy2st eval"""
        reset(self.seed)

        # data_dict = self.input_data[0]  # self.input_data.__getitem__(0)
        data_dict = self.data[0]
        net = self.net.get_layer()
        net = paddle.jit.to_static(net)
        net.eval()
        logit = net(**data_dict)
        # logit = self.loss_info.get_loss(logit)
        return logit
