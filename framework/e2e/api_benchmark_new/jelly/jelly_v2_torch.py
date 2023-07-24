#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jelly_2 用于单个产品执行
"""
import random
import timeit
import os
import json
from inspect import isclass
import torch
import numpy as np
from paddle import to_tensor

# from utils.logger import logger
from reload_config import OPERATOR_RELOAD


TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
# 计算精度，保留6位有效数字
ACCURACY = "%.6g"


class Jelly_v2_torch(object):
    """
    compare tools
    """

    def __init__(
        self,
        api,
        logger,
        # framework="paddle",
        default_dtype="float32",
        place=None,
        card=None,
        title=None,
        # enable_backward=True,
        loops=50,
        base_times=1000,
    ):
        """

        :param paddle_api:
        :param torch_api:
        :param place:  cpu or gpu (string)
        :param card: 0 1 2 3 (int)
        :param explain: case的说明 会打印在日志中
        """
        self.seed = 33
        # self.enable_backward = enable_backward
        self.debug = True
        # self.framework = framework

        torch.set_default_dtype(TORCH_DTYPE[default_dtype])

        # 循环次数
        self.loops = loops
        # timeit 基础运行时间
        self.base_times = base_times
        # 设置logger
        self.logger = logger.get_log()

        self.dump_data = []
        self.result = {}

        # set api name
        self.result["api"] = api
        # set log file name
        self.log_file_name = title
        self.result["yaml"] = self.log_file_name
        # set Reload API DICT
        self.reload = OPERATOR_RELOAD
        self.api_str = api
        # trans "str api" to obj
        if api not in self.reload.keys():
            self.api = eval(api)
        else:
            self.api = api
        self.compare_dict = None
        self.param = dict()
        self.data = dict()
        self.method = dict()
        self.places = place
        self.card = card
        self._set_seed()
        self._set_place(self.card)

    def _set_seed(self):
        """
        init seed
        :return:
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def _set_place(self, card=None):
        """
        init place
        :return:
        """
        if self.places is None:
            if torch.cuda.is_available() is True:
                if card is None:
                    torch.device(0)
                else:
                    torch.device(card)
            else:
                self.places = "cpu"
                torch.device("cpu")
        else:
            if self.places == "cpu":
                torch.device("cpu")
            else:
                if card is None:
                    torch.device(0)
                else:
                    torch.device(card)

    def _layertypes(self, func):
        """
        define layertypes
        """
        types = {0: "func", 1: "class", 2: "reload"}
        if self.api in self.reload.keys():
            return types[2]
        # 设置函数执行方式，函数式还是声明式.
        elif isclass(func):
            return types[1]
        else:
            return types[0]

    def set_torch_param(self, inputs: dict, param: dict):
        """
        设置torch 输入参数
        """
        for key, value in inputs.items():
            # 只针对输入为list[Tensor, Tensor]
            if isinstance(value, list):
                self.data[key] = []
                for i, v in enumerate(value):
                    if self.places == "cpu":
                        if isinstance(v, (np.generic, np.ndarray)):
                            self.data[key].append(torch.tensor(v))
                        else:
                            self.data[key].append(v)
                    else:
                        if isinstance(v, (np.generic, np.ndarray)):
                            self.data[key].append(torch.tensor(v).to("cuda"))
                        else:
                            self.data[key].append(v)
                    if isinstance(v, (np.generic, np.ndarray)):
                        if (
                            self.api_str.endswith("_")
                            or (v.dtype == np.int32)
                            or (v.dtype == np.int64)
                            or (v.dtype == bool)
                        ):
                            self.data[key][i].requires_grad = False
                        else:
                            self.data[key][i].requires_grad = True
            else:
                if self.places == "cpu":
                    if isinstance(value, (np.generic, np.ndarray)):
                        self.data[key] = torch.tensor(value)
                    else:
                        self.data[key] = value
                else:
                    if isinstance(value, (np.generic, np.ndarray)):
                        self.data[key] = torch.tensor(value).to("cuda")
                    else:
                        self.data[key] = value
                if isinstance(value, (np.generic, np.ndarray)):
                    if (
                        self.api_str.endswith("_")
                        or (value.dtype == np.int32)
                        or (value.dtype == np.int64)
                        or (value.dtype == bool)
                    ):
                        self.data[key].requires_grad = False
                    else:
                        self.data[key].requires_grad = True
        for key, value in param.items():
            if isinstance(value, (np.generic, np.ndarray)):
                if self.places == "cpu":
                    self.param[key] = torch.tensor(value)
                else:
                    self.param[key] = torch.tensor(value).to("cuda")
            elif key == "device":
                if self.places == "cpu":
                    self.param[key] = value
                else:
                    self.param[key] = torch.device("cuda")
            else:
                self.param[key] = value

    def set_torch_method(self, method: dict):
        """
        设置torch调用方法method
        """
        if method is not None:
            for key, value_dict in method.items():
                self.method[key] = value_dict
                for k, v in value_dict.items():
                    # 默认传入字典时，表示传入的是一个np.ndarray并转为tensor，否则传入的不为tensor。后续需优化code
                    if isinstance(v, dict):
                        self.method[key][k] = torch.tensor(v["value"])
                    else:
                        self.method[key][k] = v

    def torch_forward(self):
        """
        torch 前向时间
        """
        forward_time_list = []
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            tmp = timeit.timeit(lambda: self.api(**input_param), number=int(0.2 * self.loops * self.base_times))  # 预热
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: self.api(**input_param), number=self.base_times)
                forward_time_list.append(forward_time)
        elif self._layertypes(self.api) == "class":
            obj = self.api(**self.param)
            # 预热
            if self.method == dict():
                tmp = timeit.timeit(lambda: obj(*self.data.values()), number=int(0.2 * self.loops * self.base_times))
            else:
                obj_method = eval("obj" + "." + list(self.method.keys())[0])
                method_params_dict = self.method[list(self.method.keys())[0]]
                tmp = timeit.timeit(
                    lambda: obj_method(**method_params_dict), number=int(0.2 * self.loops * self.base_times)
                )
            for i in range(self.loops):
                if self.method == dict():
                    forward_time = timeit.timeit(lambda: obj(*self.data.values()), number=self.base_times)
                else:
                    obj_method = eval("obj" + "." + list(self.method.keys())[0])
                    method_params_dict = self.method[list(self.method.keys())[0]]
                    forward_time = timeit.timeit(lambda: obj_method(**method_params_dict), number=self.base_times)
                # forward_time = timeit.timeit(lambda: obj(*self.data.values()), number=self.base_times)
                forward_time_list.append(forward_time)
        elif self._layertypes(self.api) == "reload":
            # 判断"reload" api中有一个输入还是两个输入
            if "y" in self.data.keys():
                x = self.data["x"]
                y = self.data["y"]
                expression = self.reload.get(self.api).format("x", "y")
            else:
                x = self.data["x"]
                expression = self.reload.get(self.api).format("x")

            def func(x, y):
                eval(expression)

            def func_x(x):
                eval(expression)

            # 预热
            if "y" in self.data.keys():
                tmp = timeit.timeit(lambda: func(x, y), number=int(0.2 * self.loops * self.base_times))  # 预热
            else:
                tmp = timeit.timeit(lambda: func_x(x), number=int(0.2 * self.loops * self.base_times))  # 预热
            for i in range(self.loops):
                if "y" in self.data.keys():
                    forward_time = timeit.timeit(lambda: func(x, y), number=self.base_times)
                else:
                    forward_time = timeit.timeit(lambda: func_x(x), number=self.base_times)
                forward_time_list.append(forward_time)
        else:
            raise AttributeError

        del tmp
        return forward_time_list

    def torch_total(self):
        """
        torch 总时间
        """
        total_time_list = []
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            res = self.api(**input_param)
            # init grad tensor
            if self.places == "gpu":
                grad_tensor = torch.ones(res.shape, dtype=res.dtype).to("cuda")
            else:
                grad_tensor = torch.ones(res.shape, dtype=res.dtype)

            def func(input_param):
                res = self.api(**input_param)
                res.backward(grad_tensor)

            tmp = timeit.timeit(lambda: func(input_param), number=int(0.2 * self.loops * self.base_times))  # 预热
            for i in range(self.loops):
                total_time = timeit.timeit(lambda: func(input_param), number=self.base_times)
                total_time_list.append(total_time)
        elif self._layertypes(self.api) == "class":
            obj = self.api(**self.param)
            if self.method == dict():
                res = obj(*self.data.values())
            else:
                obj_method = eval("obj" + "." + list(self.method.keys())[0])
                method_params_dict = self.method[list(self.method.keys())[0]]
                res = obj_method(**method_params_dict)
            # res = obj(*self.data.values())
            # init grad tensor
            if self.places == "gpu":
                grad_tensor = torch.ones(res.shape, dtype=res.dtype).to("cuda")
            else:
                grad_tensor = torch.ones(res.shape, dtype=res.dtype)

            def clas(input_param):
                """lambda clas"""
                res = obj(*input_param)
                res.backward(grad_tensor)

            def clas_method(input_param):
                res = obj_method(**input_param)
                res.backward(grad_tensor)

            # 预热
            if self.method == dict():
                tmp = timeit.timeit(lambda: clas(self.data.values()), number=int(0.2 * self.loops * self.base_times))
            else:
                tmp = timeit.timeit(
                    lambda: clas_method(method_params_dict), number=int(0.2 * self.loops * self.base_times)
                )
            for i in range(self.loops):
                if self.method == dict():
                    total_time = timeit.timeit(lambda: clas(self.data.values()), number=self.base_times)
                else:
                    total_time = timeit.timeit(lambda: clas_method(method_params_dict), number=self.base_times)
                total_time_list.append(total_time)
        elif self._layertypes(self.api) == "reload":
            # 判断"reload" api中有一个输入还是两个输入
            if "y" in self.data.keys():
                x = self.data["x"]
                y = self.data["y"]
                expression = self.reload.get(self.api).format("x", "y")
            else:
                x = self.data["x"]
                expression = self.reload.get(self.api).format("x")
            res = eval(expression)
            if self.places == "gpu":
                grad_tensor = torch.ones(res.shape, dtype=res.dtype).to("cuda")
            else:
                grad_tensor = torch.ones(res.shape, dtype=res.dtype)

            def func(x, y):
                res = eval(expression)
                res.backward(grad_tensor)

            def func_x(x):
                res = eval(expression)
                res.backward(grad_tensor)

            # 预热
            if "y" in self.data.keys():
                tmp = timeit.timeit(lambda: func(x, y), number=int(0.2 * self.loops * self.base_times))  # 预热
            else:
                tmp = timeit.timeit(lambda: func_x(x), number=int(0.2 * self.loops * self.base_times))  # 预热
            for i in range(self.loops):
                if "y" in self.data.keys():
                    total_time = timeit.timeit(lambda: func(x, y), number=self.base_times)
                else:
                    total_time = timeit.timeit(lambda: func_x(x), number=self.base_times)
                total_time_list.append(total_time)
        else:
            raise AttributeError

        del tmp
        return total_time_list

    def _save(self, data):
        """
        保存数据到磁盘
        :return:
        """
        log_file = "./log/{}.json".format(self.log_file_name)
        if not os.path.exists("./log"):
            os.makedirs("./log")
        try:
            with open(log_file, "w") as json_file:
                json.dump(data, json_file)
            self.logger.info("[{}] log file save success!".format(self.log_file_name))
        except Exception as e:
            print(e)
