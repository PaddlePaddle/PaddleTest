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
            if isinstance(v, dict):
                if "generate_way" in v and "type" in v:
                    if v["generate_way"] == "random":
                        if v["type"] == "numpy":
                            value = tool._randtool(
                                dtype=v["dtype"], low=v["range"][0], high=v["range"][1], shape=v["shape"]
                            )
                            paddle_data_dict[k] = value
                        elif v["type"] == "Tensor":
                            value = paddle.to_tensor(
                                tool._randtool(
                                    dtype=v["dtype"], low=v["range"][0], high=v["range"][1], shape=v["shape"]
                                )
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
                            value = paddle.to_tensor(value, dtype=v["dtype"])
                            paddle_data_dict[k] = value
                        elif v["type"] == "int" or v["type"] == "float":
                            paddle_data_dict[k] = value
                    elif v["generate_way"] == "load":
                        self.logger.get_log().error("暂未开发加载路径下数据！！！~~~")
                else:
                    paddle_data_dict[k] = {}
                    for j, w in v.items():
                        if isinstance(w, list) or isinstance(w, tuple):
                            paddle_data_dict[k][j] = []
                            for u in w:
                                if u["generate_way"] == "random":
                                    if u["type"] == "numpy":
                                        value = tool._randtool(
                                            dtype=u["dtype"], low=u["range"][0], high=u["range"][1], shape=u["shape"]
                                        )
                                        paddle_data_dict[k][j].append(value)
                                    elif u["type"] == "Tensor":
                                        value = paddle.to_tensor(
                                            tool._randtool(
                                                dtype=u["dtype"],
                                                low=u["range"][0],
                                                high=u["range"][1],
                                                shape=u["shape"],
                                            )
                                        )
                                        paddle_data_dict[k][j].append(value)
                                    else:
                                        self.logger.get_log().error(
                                            "yaml格式不规范: input为random随机时, 输入类型不可为{}".format(u["type"])
                                        )
                                elif u["generate_way"] == "solid":
                                    value = u["value"]
                                    if u["type"] == "numpy":
                                        value = np.array(value).astype(u["dtype"])
                                        paddle_data_dict[k][j].append(value)
                                    elif u["type"] == "Tensor":
                                        value = paddle.to_tensor(value, dtype=u["dtype"])
                                        paddle_data_dict[k][j].append(value)
                                    elif u["type"] == "int" or u["type"] == "float":
                                        paddle_data_dict[k][j].append(value)
                                elif u["generate_way"] == "load":
                                    self.logger.get_log().error("暂未开发加载路径下数据！！！~~~")
                        else:
                            if w["generate_way"] == "random":
                                if w["type"] == "numpy":
                                    value = tool._randtool(
                                        dtype=w["dtype"], low=w["range"][0], high=w["range"][1], shape=w["shape"]
                                    )
                                    paddle_data_dict[k][j] = value
                                elif w["type"] == "Tensor":
                                    value = paddle.to_tensor(
                                        tool._randtool(
                                            dtype=w["dtype"], low=w["range"][0], high=w["range"][1], shape=w["shape"]
                                        )
                                    )
                                    paddle_data_dict[k][j] = value
                                else:
                                    self.logger.get_log().error(
                                        "yaml格式不规范: input为random随机时, 输入类型不可为{}".format(w["type"])
                                    )
                            elif w["generate_way"] == "solid":
                                value = w["value"]
                                if w["type"] == "numpy":
                                    value = np.array(value).astype(w["dtype"])
                                    paddle_data_dict[k][j] = value
                                elif w["type"] == "Tensor":
                                    value = paddle.to_tensor(value, dtype=w["dtype"])
                                    paddle_data_dict[k][j] = value
                                elif w["type"] == "int" or w["type"] == "float":
                                    paddle_data_dict[k][j] = value
                            elif w["generate_way"] == "load":
                                self.logger.get_log().error("暂未开发加载路径下数据！！！~~~")

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
                            self.logger.get_log().error("yaml格式不规范: input为random随机时, 输入类型不可为{}".format(i["type"]))
                    elif i["generate_way"] == "solid":
                        value = i["value"]
                        if i["type"] == "numpy":
                            value = np.array(value).astype(i["dtype"])
                            paddle_data_dict[k].append(value)
                        elif i["type"] == "Tensor":
                            value = paddle.to_tensor(value, dtype=i["dtype"])
                            paddle_data_dict[k].append(value)
                        elif i["type"] == "int" or i["type"] == "float" or i["type"] == "list":
                            paddle_data_dict[k].append(value)
                    elif i["generate_way"] == "load":
                        self.logger.get_log().error("暂未开发加载路径下数据！！！~~~")
                if isinstance(v, tuple):
                    paddle_data_dict[k] = tuple(paddle_data_dict[k])

        paddle_data = eval(self.data_module)(paddle_data_dict)
        return paddle_data

    def get_model_dtype(self):
        """get first data type"""
        for k, v in self.data.items():
            if isinstance(v, dict):
                if "generate_way" in v and "type" in v:
                    # print('v is', v)
                    if v["dtype"] == "float64":
                        model_dtype = "float64"
                        return model_dtype
                    else:
                        model_dtype = "float32"
                else:
                    for j, w in v.items():
                        if isinstance(w, list) or isinstance(w, tuple):
                            for l in w:
                                if l["dtype"] == "float64":
                                    model_dtype = "float64"
                                    return model_dtype
                                else:
                                    model_dtype = "float32"
                        else:
                            if w["dtype"] == "float64":
                                model_dtype = "float64"
                                return model_dtype
                            else:
                                model_dtype = "float32"
            elif isinstance(v, list) or isinstance(v, tuple):
                for i in v:
                    if i["dtype"] == "float64":
                        model_dtype = "float64"
                        return model_dtype
                    else:
                        model_dtype = "float32"
        return model_dtype

    def get_single_inputspec(self):
        """get single inputspec"""
        spec_list = []
        for k, v in self.data.items():
            if v["type"] == "Tensor":
                spec_tmp = paddle.static.InputSpec(shape=v["shape"], dtype=v["dtype"], name=k)
                spec_list.append(spec_tmp)
        return spec_list
