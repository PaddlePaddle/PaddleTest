#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""

import numpy as np
import paddle
import diy
import tool


class BuildData(object):
    """BuildData"""

    def __init__(self, data_info):
        """init"""
        self.data_module = data_info["DataGenerator_name"]
        self.data = data_info["data"]

    def get_single_data(self):
        """get data"""
        paddle_data_dict = {}
        for k, v in self.data.items():
            if v["generate_way"] == "random":
                if v["type"] == "numpy":
                    value = tool._randtool(dtype=v["dtype"], low=v["range"][0], high=v["range"][1], shape=v["shape"])
                    paddle_data_dict[k] = value
                elif v["type"] == "Tensor":
                    value = paddle.to_tensor(
                        tool._randtool(dtype=v["dtype"], low=v["range"][0], high=v["range"][1], shape=v["shape"])
                    )
                    paddle_data_dict[k] = value
                else:
                    self.logger.get_log().error("yaml格式不规范: input为random随机时, 输入类型不可为{}".format(v["type"]))
            elif v["generate_way"] == "solid":
                value = v["value"]
                if v["type"] == "numpy":
                    value = np.array(value).astype(v["dtype"])
                    paddle_data_dict[k] = value
                elif v["type"] == "Tensor":
                    value = paddle.to_tensor(value)
                    paddle_data_dict[k] = value
            elif v["generate_way"] == "load":
                self.logger.get_log().error("暂未开发加载路径下数据！！！~~~")

        paddle_data = eval(self.data_module)(paddle_data_dict)

        # # data_module_type用于标识区别Dataset和DataLoader
        # data_module_type = 'Dataset'
        # if isinstance(paddle_data, Iterable):

        return paddle_data

    def get_single_inputspec(self):
        """get single inputspec"""
        spec_list = []
        for k, v in self.data.items():
            if v["type"] == "Tensor":
                spec_tmp = paddle.static.InputSpec(shape=v["shape"], dtype=v["dtype"], name=k)
                spec_list.append(spec_tmp)
        return spec_list
