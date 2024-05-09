#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""

import os
import itertools
import numpy as np

if os.environ.get("FRAMEWORK") == "paddle":
    import paddle
    import diy
    import layerApicase
    import layercase
elif os.environ.get("FRAMEWORK") == "torch":
    import torch
    import layerTorchcase

import tools.np_tool as tool


class BuildData(object):
    """BuildData"""

    def __init__(self, layerfile):
        """init"""
        self.dataname = layerfile

    def get_single_data(self):
        """get data"""
        dataname = self.dataname + ".create_numpy_inputs()"
        data = []
        for i in eval(dataname):
            if os.environ.get("FRAMEWORK") == "paddle":
                data.append(paddle.to_tensor(i, stop_gradient=False))
            elif os.environ.get("FRAMEWORK") == "torch":
                data.append(torch.tensor(i, requires_grad=True))

        return data

    def get_single_tensor(self):
        """get data"""
        dataname = self.dataname + ".create_tensor_inputs()"
        data = []
        for i in eval(dataname):
            data.append(i)

        return data

    def get_single_numpy(self):
        """get data"""
        dataname = self.dataname + ".create_numpy_inputs()"
        data = []
        for i in eval(dataname):
            data.append(i)

        return data

    def get_single_input_and_spec(self):
        """get single inputspec"""
        spec_list = []
        data = self.get_single_data()
        for v in data:
            if isinstance(v, paddle.Tensor):
                input_shape = tuple([-1] * len(v.shape))
                spec_tmp = paddle.static.InputSpec(
                    shape=input_shape, dtype=v.dtype, name=None, stop_gradient=v.stop_gradient
                )
                spec_list.append(spec_tmp)
        return data, spec_list

    def get_single_input_and_multi_spec(self):
        """get single inputspec"""
        spec_all_list = []
        data = self.get_single_data()
        for index in len(data):
            spec_list = []
            maybe_shapes = self._input_spec_shape_search(data[index])
            for i, v in enumerate(data):
                for j, w in enumerate(maybe_shapes):
                    if isinstance(v, paddle.Tensor):
                        if i == index:
                            input_shape = w
                        else:
                            input_shape = tuple([-1] * len(v.shape))
                        spec_tmp = paddle.static.InputSpec(
                            shape=input_shape, dtype=v.dtype, name=None, stop_gradient=v.stop_gradient
                        )
                        spec_list.append(spec_tmp)
            spec_all_list.append(spec_list)
        return data, spec_all_list

    def _input_spec_shape_search(self, shape):
        """用于搜索inputspect可能的shape组合"""
        shape_eles = [[-1, s] for s in shape]
        raw_shapes = itertools.product(*shape_eles)
        has_channel = len(shape) == 4 and shape[1] == 3

        maybe_shapes = []
        for s in raw_shapes:
            # in case of channel
            if has_channel and s[1] is None:
                continue
            maybe_shapes.append(s)
        return maybe_shapes
