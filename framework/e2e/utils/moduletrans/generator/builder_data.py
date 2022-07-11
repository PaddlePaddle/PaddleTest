#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""
import numpy as np
import paddle
import tool


class BuildData(object):
    """BuildData"""

    def __init__(self, data_loader, data):
        """init"""
        self.data_loader = data_loader
        self.data = data

    def get_single_numpy_data(self):
        """get data"""
        np_data = {}
        for k, v in self.data.items():
            if v["input"]["random"]:
                value = tool._randtool(
                    dtype=v["input"]["dtype"],
                    low=v["input"]["range"][0],
                    high=v["input"]["range"][1],
                    shape=v["input"]["shape"],
                )
                np_data[k] = value
            else:
                np_data[k] = v["input"]["value"]
        return np_data

    def get_single_paddle_data(self):
        """get data"""
        paddle_data = {}
        for k, v in self.data.items():
            if v["input"]["random"]:
                if v["input"]["type"] == "numpy":
                    value = tool._randtool(
                        dtype=v["input"]["dtype"],
                        low=v["input"]["range"][0],
                        high=v["input"]["range"][1],
                        shape=v["input"]["shape"],
                    )
                    paddle_data[k] = value
                elif v["input"]["type"] == "Tensor":
                    value = paddle.to_tensor(
                        tool._randtool(
                            dtype=v["input"]["dtype"],
                            low=v["input"]["range"][0],
                            high=v["input"]["range"][1],
                            shape=v["input"]["shape"],
                        )
                    )
                    paddle_data[k] = value
                else:
                    self.logger.get_log().error("yaml格式不规范: input为random随机时, 输入类型不可为{}".format(v["input"]["type"]))
            else:
                value = v["input"]["value"]
                if v["input"]["type"] == "numpy":
                    value = np.array(value).astype(v["input"]["dtype"])
                elif v["input"]["type"] == "Tensor":
                    value = paddle.to_tensor(value)
                paddle_data[k] = value
        return paddle_data

    def get_single_inputspec(self):
        """get single inputspec"""
        spec_list = []
        for k, v in self.data.items():
            if v["input"]["type"] == "Tensor":
                spec_tmp = paddle.static.InputSpec(shape=v["input"]["shape"], dtype=v["input"]["dtype"], name=k)
                spec_list.append(spec_tmp)
        return spec_list
