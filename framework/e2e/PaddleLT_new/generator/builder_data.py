#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""

import os
import numpy as np
import paddle
import diy
import layercase

# import tool
import tools.np_tool as tool


class BuildData(object):
    """BuildData"""

    def __init__(self, layerfile):
        """init"""
        # self.data_module = data_info["DataGenerator_name"]
        self.dataname = layerfile
        # self.dtype = data_type

    def get_single_data(self):
        """get data"""
        dataname = self.dataname + ".create_numpy_inputs()"
        data = []
        for i in eval(dataname):
            data.append(paddle.to_tensor(i))

        return data

    def get_single_tensor(self):
        """get data"""
        dataname = self.dataname + ".create_paddle_inputs()"
        data = []
        for i in eval(dataname):
            data.append(paddle.to_tensor(i))

        return data

    # def get_single_inputspec(self):
    #     """get single inputspec"""
    #     spec_list = []
    #     for k, v in self.data.items():
    #         if v["type"] == "Tensor":
    #             spec_tmp = paddle.static.InputSpec(shape=v["shape"], dtype=v["dtype"], name=k)
    #             spec_list.append(spec_tmp)
    #     return spec_list
