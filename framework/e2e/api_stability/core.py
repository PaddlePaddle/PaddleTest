#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
稳定性执行类
"""

from inspect import isclass
import random
import numpy as np
import paddle
from paddle import to_tensor
from utils.logger import logger
from copy import deepcopy



class Core(object):
    """
    稳定性执行类
    """

    def __init__(self, paddle_api, dtype="float64"):
        self.seed = 33
        self.enable_backward = True
        self.debug = False
        self.api = paddle_api
        self.compare_dict = None
        self.places = None
        self._set_seed()
        self.ignore_var = []
        self.types = None
        self.data = dict()
        self.param = dict()
        self.forward_res = []
        self.grad_res = []
        self.inputs = dict()
        self.loops = 1000
        self.hook()
        paddle.set_default_dtype(dtype)


    def hook(self):
        """
        hook
        """
        pass

    def _set_seed(self):
        """
        init seed
        :return:
        """
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        random.seed(self.seed)

    def _layertypes(self, func):
        """
        define layertypes
        """
        types = {0: "func", 1: "class"}
        # 设置函数执行方式，函数式还是声明式.
        if isclass(func):
            return types[1]
        else:
            return types[0]

    def set_paddle_param(self, inputs: dict, param: dict):
        """
        设置paddle 输入参数
        """
        for key, value in inputs.items():
            # 只针对输入为list[Tensor, Tensor]
            if isinstance(value, list):
                self.data[key] = []
                for i, v in enumerate(value):
                    if isinstance(v, (np.generic, np.ndarray)):
                        # self.logger.info("v.dtype is : {}".format(v.dtype))
                        self.data[key].append(to_tensor(v))
                        if (
                            self.api.endswith("_")
                            or (v.dtype == np.int32)
                            or (v.dtype == np.int64)
                            or (v.dtype == bool)
                        ):
                            self.data[key][i].stop_gradient = True
                        else:
                            self.data[key][i].stop_gradient = False
                    else:
                        self.data[key].append(v)
                    # self.logger.info("self.data[key].stop_gradient is : {}".format(self.data[key][i].stop_gradient))
            else:
                if isinstance(value, (np.generic, np.ndarray)):
                    # self.logger.info("value.dtype is : {}".format(value.dtype))
                    self.data[key] = to_tensor(value)
                    if (
                        self.api.endswith("_")
                        or (value.dtype == np.int32)
                        or (value.dtype == np.int64)
                        or (value.dtype == bool)
                    ):
                        self.data[key].stop_gradient = True
                    else:
                        self.data[key].stop_gradient = False
                else:
                    self.data[key] = value
                # self.logger.info("self.data[key].stop_gradient is : {}".format(self.data[key].stop_gradient))
        for key, value in param.items():
            if isinstance(value, (np.generic, np.ndarray)):
                self.param[key] = to_tensor(value)
            else:
                self.param[key] = value

    def paddle_run(self):
        """
        计算paddle 总体时间
        """
        self.api = eval(self.api)
        if self._layertypes(self.api) == "func":
            for i in range(self.loops):
                input_param = dict(self.data, **self.param)
                res = self.api(**input_param)
                grad = paddle.grad([res], *self.data.values(), retain_graph=False)
                self.forward_res.append(res.numpy())
                self.grad_res.append(grad)
        elif self._layertypes(self.api) == "class":
            obj = self.api(**self.param)
            for i in range(self.loops):
                res = obj(*self.data.values())
                grad = paddle.grad([res], *self.data.values(), retain_graph=False)
                self.forward_res.append(res.numpy())
                self.grad_res.append(grad)
        else:
            raise AttributeError
        return self.forward_res, self.grad_res