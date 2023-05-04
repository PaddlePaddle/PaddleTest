#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nn.Layer配置相关
"""

import os
import numpy as np
import paddle
import tool


class LayerTrans(object):
    """获取单个case中的Layer参数信息"""

    def __init__(self, case):
        """init"""
        self.case = case

    def get_layer(self):
        """get layer"""
        layer_name = self.case.get("Layer").get("layer_name")
        layer_param = self.case.get("Layer").get("params")
        # return repo, layer_name, layer_param
        # return layer_name, layer_param

        if layer_param is not None:
            # 带有**标记的字符串转换为python object
            for k, v in layer_param.items():
                if isinstance(v, str):
                    if "**" in v:
                        try:
                            tmp = v
                            tmp = tmp.replace("**", "")
                            layer_param[k] = eval(tmp)
                        except:
                            layer_param[k] = v

        if layer_param is not None:
            layer = eval(layer_name)(**layer_param)
        else:
            layer = eval(layer_name)()
        return layer

    def get_paddle_data(self):
        """get paddle data"""
        data_module = self.case.get("DataGenerator").get("DataGenerator_name")
        data_info = self.case.get("DataGenerator").get("data")
        paddle_data_dict = {}
        for k, v in data_info.items():
            if isinstance(v, dict):
                if v["generate_way"] == "random":
                    if v["type"] == "numpy":
                        value = tool._randtool(
                            dtype=v["dtype"], low=v["range"][0], high=v["range"][1], shape=v["shape"]
                        )
                        paddle_data_dict[k] = value
                    elif v["type"] == "Tensor":
                        value = paddle.to_tensor(
                            tool._randtool(dtype=v["dtype"], low=v["range"][0], high=v["range"][1], shape=v["shape"])
                        )
                        paddle_data_dict[k] = value
                    else:
                        raise Exception("yaml格式不规范: input为random随机时, 输入类型不可为{}".format(v["type"]))
                elif v["generate_way"] == "solid":
                    value = v["value"]
                    if v["type"] == "numpy":
                        value = np.array(value).astype(v["dtype"])
                        paddle_data_dict[k] = value
                    elif v["type"] == "Tensor":
                        value = paddle.to_tensor(value)
                        paddle_data_dict[k] = value
                    elif v["type"] == "int" or v["type"] == "float":
                        paddle_data_dict[k] = value
                elif v["generate_way"] == "load":
                    raise Exception("暂未开发加载路径下数据！！！~~~")

            elif isinstance(v, list) or isinstance(v, tuple):
                paddle_data_dict[k] = []
                for i in v:
                    if i["generate_way"] == "random":
                        if i["type"] == "numpy":
                            value = tool._randtool(
                                dtype=i["dtype"], low=i["range"][0], high=i["range"][1], shape=i["shape"]
                            )
                            paddle_data_dict[k].append(value)
                        elif i["type"] == "Tensor":
                            value = paddle.to_tensor(
                                tool._randtool(
                                    dtype=i["dtype"], low=i["range"][0], high=i["range"][1], shape=i["shape"]
                                )
                            )
                            paddle_data_dict[k].append(value)
                        else:
                            raise Exception("yaml格式不规范: input为random随机时, 输入类型不可为{}".format(i["type"]))
                    elif i["generate_way"] == "solid":
                        value = i["value"]
                        if i["type"] == "numpy":
                            value = np.array(value).astype(i["dtype"])
                            paddle_data_dict[k].append(value)
                        elif i["type"] == "Tensor":
                            value = paddle.to_tensor(value)
                            paddle_data_dict[k].append(value)
                        elif i["type"] == "int" or i["type"] == "float":
                            paddle_data_dict[k].append(value)
                    elif i["generate_way"] == "load":
                        raise Exception("暂未开发加载路径下数据！！！~~~")
                if isinstance(v, tuple):
                    paddle_data_dict[k] = tuple(paddle_data_dict[k])

        paddle_data = eval(data_module)(paddle_data_dict)
        return paddle_data
