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

    # def __init__(self, testing, case, layer):
    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.modelpath = layerfile.replace(".py", "").rsplit(".", 1)[0].replace(".", "/")
        self.layername = layerfile.replace(".py", "").rsplit(".", 1)[1].replace(".", "/")

        self.testing = testing

        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.net = BuildLayer(layerfile=layerfile).get_layer()

        self.data = BuildData(layerfile=layerfile).get_single_data()

        self.path = os.path.join(os.getcwd(), "test_prodct", self.modelpath)

    def jit_save(self):
        """jit.save(layer)"""
        reset(self.seed)

        net = paddle.jit.to_static(self.net)
        net.eval()
        net(*self.data)

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(net, path=os.path.join(self.path, self.layername))
