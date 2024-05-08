#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""

import os
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
