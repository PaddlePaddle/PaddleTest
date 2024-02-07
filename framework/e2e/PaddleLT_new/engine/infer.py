#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
infer 执行器
"""

import os
import numpy as np
import paddle
import paddle.inference as paddle_infer
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class LayerInfer(object):
    """
    构建Layer预测的通用类
    """

    def __init__(self, testing, layerfile):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)
        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.set_device(str(self.device))
        self.device_id = os.environ.get("PLT_DEVICE_ID")

        self.testing = testing

        self.data = BuildData(layerfile=layerfile).get_single_numpy()

        self.path = os.path.join(os.getcwd(), "test_prodct", layerfile.replace(".", "/"))

    def paddle_infer_gpu(self):
        """infer load (layer)"""
        reset(self.seed)
        if not os.path.exists(self.path + ".pdiparams"):
            return "pass"

        config = paddle_infer.Config(self.path + ".pdmodel", self.path + ".pdiparams")
        config.enable_use_gpu(1000, int(self.device_id))

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            input_tmp = self.data[i]
            input_handle.copy_from_cpu(input_tmp)

        predictor.run()
        output_names = predictor.get_output_names()
        if len(output_names) > 1:
            infer_res = []
            for i, name in enumerate(output_names):
                output_handle = predictor.get_output_handle(output_names[i])
                infer_res.append(output_handle.copy_to_cpu())
        else:
            output_handle = predictor.get_output_handle(output_names[0])
            infer_res = output_handle.copy_to_cpu()
        return infer_res

    def paddle_infer_cpu(self):
        """infer load (layer)"""
        reset(self.seed)
        if not os.path.exists(self.path + ".pdiparams"):
            return "pass"

        config = paddle_infer.Config(self.path + ".pdmodel", self.path + ".pdiparams")

        config.disable_mkldnn()

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            input_tmp = self.data[i]
            input_handle.copy_from_cpu(input_tmp)

        predictor.run()
        output_names = predictor.get_output_names()
        if len(output_names) > 1:
            infer_res = []
            for i, name in enumerate(output_names):
                output_handle = predictor.get_output_handle(output_names[i])
                infer_res.append(output_handle.copy_to_cpu())
        else:
            output_handle = predictor.get_output_handle(output_names[0])
            infer_res = output_handle.copy_to_cpu()
        return infer_res

    def paddle_infer_mkldnn(self):
        """infer load (layer)"""
        reset(self.seed)
        if not os.path.exists(self.path + ".pdiparams"):
            return "pass"

        config = paddle_infer.Config(self.path + ".pdmodel", self.path + ".pdiparams")

        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(1)
        config.set_mkldnn_cache_capacity(1)

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            input_tmp = self.data[i]
            input_handle.copy_from_cpu(input_tmp)

        predictor.run()
        output_names = predictor.get_output_names()
        if len(output_names) > 1:
            infer_res = []
            for i, name in enumerate(output_names):
                output_handle = predictor.get_output_handle(output_names[i])
                infer_res.append(output_handle.copy_to_cpu())
        else:
            output_handle = predictor.get_output_handle(output_names[0])
            infer_res = output_handle.copy_to_cpu()
        return infer_res

    def paddle_infer_ort(self):
        """infer load (layer)"""
        reset(self.seed)
        if not os.path.exists(self.path + ".pdiparams"):
            return "pass"

        config = paddle_infer.Config(self.path + ".pdmodel", self.path + ".pdiparams")

        config.enable_onnxruntime()
        config.enable_ort_optimization()

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            input_tmp = self.data[i]
            input_handle.copy_from_cpu(input_tmp)

        predictor.run()
        output_names = predictor.get_output_names()
        if len(output_names) > 1:
            infer_res = []
            for i, name in enumerate(output_names):
                output_handle = predictor.get_output_handle(output_names[i])
                infer_res.append(output_handle.copy_to_cpu())
        else:
            output_handle = predictor.get_output_handle(output_names[0])
            infer_res = output_handle.copy_to_cpu()
        return infer_res
