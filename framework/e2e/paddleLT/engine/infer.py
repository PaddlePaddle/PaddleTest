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
import paddle.inference as paddle_infer
from engine.xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData


class LayerInfer(object):
    """
    构建Layer预测的通用类
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

        self.data_info = self.layer.get("DataGenerator")
        self.data = BuildData(data_info=self.data_info).get_single_data()

        self.path = os.path.join(os.getcwd(), "test_prodct", *self.layer_name.split("."))

    def paddle_infer(self):
        """infer load (layer)"""
        reset(self.seed)

        config = paddle_infer.Config(
            os.path.join(self.path, "{}.pdmodel".format(self.case)),
            os.path.join(self.path, "{}.pdiparams".format(self.case)),
        )

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()

        ##
        input_dict = self.data.__getitem__(0)
        inputs_key = sorted(input_dict.keys())

        input_list = []
        for k in inputs_key:
            input_list.append(input_dict[k])

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            if isinstance(input_list[i], int):
                input_tmp = np.array(input_list[i])
            elif isinstance(input_list[i], paddle.Tensor):
                input_tmp = input_list[i].numpy()

            input_handle.copy_from_cpu(input_tmp)

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        infer_res = output_handle.copy_to_cpu()
        return infer_res

    def paddle_infer_new(self):
        """infer load (layer)"""
        reset(self.seed)

        config = paddle_infer.Config(
            os.path.join(self.path, "{}.pdmodel".format(self.case)),
            os.path.join(self.path, "{}.pdiparams".format(self.case)),
        )

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()

        ##
        input_dict = self.data.__getitem__(0)
        inputs_key = sorted(input_dict.keys())

        input_list = []
        for k in inputs_key:
            input_list.append(input_dict[k])

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            if isinstance(input_list[i], int):
                input_tmp = np.array(input_list[i])
            elif isinstance(input_list[i], paddle.Tensor):
                input_tmp = input_list[i].numpy()

            input_handle.copy_from_cpu(input_tmp)

        predictor.run()

        # output_data_dict = {}
        # output_names = predictor.get_output_names()
        # for _, output_data_name in enumerate(output_names):
        #     output_handle = predictor.get_output_handle(output_data_name)
        #     output_data = output_handle.copy_to_cpu()
        #     output_data_dict[output_data_name] = output_data

        output_data_list = []
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_list.append(output_data)

        return output_data_list

    def paddle_infer_mkldnn(self):
        """infer load (layer)"""
        reset(self.seed)

        config = paddle_infer.Config(
            os.path.join(self.path, "{}.pdmodel".format(self.case)),
            os.path.join(self.path, "{}.pdiparams".format(self.case)),
        )

        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(1)
        config.set_mkldnn_cache_capacity(1)

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()

        ##
        input_dict = self.data.__getitem__(0)
        inputs_key = sorted(input_dict.keys())

        input_list = []
        for k in inputs_key:
            input_list.append(input_dict[k])

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            if isinstance(input_list[i], int):
                input_tmp = np.array(input_list[i])
            elif isinstance(input_list[i], paddle.Tensor):
                input_tmp = input_list[i].numpy()

            input_handle.copy_from_cpu(input_tmp)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        infer_res = output_handle.copy_to_cpu()
        return infer_res

    def paddle_infer_ort(self):
        """infer load (layer)"""
        reset(self.seed)

        config = paddle_infer.Config(
            os.path.join(self.path, "{}.pdmodel".format(self.case)),
            os.path.join(self.path, "{}.pdiparams".format(self.case)),
        )

        config.enable_onnxruntime()
        config.enable_ort_optimization()

        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()

        ##
        input_dict = self.data.__getitem__(0)
        inputs_key = sorted(input_dict.keys())

        input_list = []
        for k in inputs_key:
            input_list.append(input_dict[k])

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            if isinstance(input_list[i], int):
                input_tmp = np.array(input_list[i])
            elif isinstance(input_list[i], paddle.Tensor):
                input_tmp = input_list[i].numpy()

            input_handle.copy_from_cpu(input_tmp)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        infer_res = output_handle.copy_to_cpu()
        return infer_res
