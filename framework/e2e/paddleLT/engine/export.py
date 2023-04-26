#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jit 方法
"""
import os
import numpy as np
import paddle
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class LayerExport(object):
    """
    构建Layer导出的通用类
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

        self.path = os.path.join(os.getcwd(), "test_prodct", *self.layer_name.split("."))

        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

    def jit_save(self):
        """jit.save(layer)"""
        reset(self.seed)

        net = self.net.get_layer()
        net = paddle.jit.to_static(net)
        net.eval()
        data_dict = self.data[0]
        net(**data_dict)
        # paddle.jit.save(net, path=os.path.join(self.save_path, "jit_save", self.case_name))
        paddle.jit.save(net, path=os.path.join(self.path, self.case))
