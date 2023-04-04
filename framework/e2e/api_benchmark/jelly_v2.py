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
import paddle
import numpy as np
from paddle import to_tensor
from utils.logger import logger
from reload_config import OPERATOR_RELOAD


PADDLE_DTYPE = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
# 计算精度，保留6位有效数字
ACCURACY = "%.6g"


class Jelly_v2(object):
    """
    compare tools
    """

    def __init__(
        self, api, framework="paddle", default_dtype="float32", place=None, card=None, title=None, enable_backward=True
    ):
        """

        :param paddle_api:
        :param torch_api:
        :param place:  cpu or gpu (string)
        :param card: 0 1 2 3 (int)
        :param explain: case的说明 会打印在日志中
        """
        self.seed = 33
        self.enable_backward = enable_backward
        self.debug = True
        self.framework = framework

        paddle.set_default_dtype(PADDLE_DTYPE[default_dtype])
        # 循环次数
        self.loops = 50
        # timeit 基础运行时间
        self.base_times = 1000
        # 设置logger
        self.logger = logger.get_log()

        self.forward_time = []
        self.backward_time = []
        self.total_time = []

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
        paddle.seed(self.seed)
        random.seed(self.seed)

    def _set_place(self, card=None):
        """
        init place
        :return:
        """
        if self.places is None:
            if paddle.is_compiled_with_cuda() is True:
                paddle.set_device("gpu:0")
            else:
                self.places = "cpu"
                paddle.set_device("cpu")
        else:
            if self.places == "cpu":
                paddle.set_device("cpu")
            else:
                if card is None:
                    paddle.set_device("gpu:0")
                else:
                    paddle.set_device("gpu:{}".format(card))

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

    # def baseline(self):
    #     print(timeit.timeit('"-".join(str(n) for n in range(100))', number=10000))

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
                            self.api_str.endswith("_")
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
                        self.api_str.endswith("_")
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

    def set_paddle_method(self, method: dict):
        """
        设置paddle调用方法method
        """
        if method is not None:
            for key, value_dict in method.items():
                self.method[key] = value_dict
                for k, v in value_dict.items():
                    # 默认传入字典时，表示传入的是一个np.ndarray并转为tensor，否则传入的不为tensor。后续需优化code
                    if isinstance(v, dict):
                        self.method[key][k] = to_tensor(v["value"])
                    else:
                        self.method[key][k] = v

    def paddle_forward(self):
        """
        主体测试逻辑
        """
        # print("self.data is: ", self.data)
        # print("self.param is: ", self.param)
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: self.api(**input_param), number=self.base_times)
                self.forward_time.append(forward_time)
        elif self._layertypes(self.api) == "class":
            # obj = self.api(**self.param)
            # for i in range(self.loops):
            #     forward_time = timeit.timeit(lambda: obj(*self.data.values()), number=self.base_times)
            #     self.forward_time.append(forward_time)
            obj = self.api(**self.param)
            for i in range(self.loops):
                if self.method == dict():
                    forward_time = timeit.timeit(lambda: obj(*self.data.values()), number=self.base_times)
                else:
                    obj_method = eval("obj" + "." + list(self.method.keys())[0])
                    method_params_dict = self.method[list(self.method.keys())[0]]
                    forward_time = timeit.timeit(lambda: obj_method(**method_params_dict), number=self.base_times)
                self.forward_time.append(forward_time)
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

            for i in range(self.loops):
                if "y" in self.data.keys():
                    forward_time = timeit.timeit(lambda: func(x, y), number=self.base_times)
                else:
                    forward_time = timeit.timeit(lambda: func_x(x), number=self.base_times)
                self.forward_time.append(forward_time)
        else:
            raise AttributeError

    def paddle_total(self):
        """
        计算paddle 总体时间
        """
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            res = self.api(**input_param)
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def func(input_param):
                res = self.api(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: func(input_param), number=self.base_times)
                self.total_time.append(total_time)
        elif self._layertypes(self.api) == "class":
            # obj = self.api(**self.param)
            # res = obj(*self.data.values())
            # grad_tensor = paddle.ones(res.shape, res.dtype)
            obj = self.api(**self.param)
            if self.method == dict():
                res = obj(*self.data.values())
            else:
                obj_method = eval("obj" + "." + list(self.method.keys())[0])
                method_params_dict = self.method[list(self.method.keys())[0]]
                res = obj_method(**method_params_dict)
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def clas(input_param):
                res = obj(*input_param)
                res.backward(grad_tensor)

            def clas_method(input_param):
                res = obj_method(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                if self.method == dict():
                    total_time = timeit.timeit(lambda: clas(self.data.values()), number=self.base_times)
                else:
                    total_time = timeit.timeit(lambda: clas_method(method_params_dict), number=self.base_times)
                self.total_time.append(total_time)
        elif self._layertypes(self.api) == "reload":
            if "y" in self.data.keys():
                x = self.data["x"]
                y = self.data["y"]
                expression = self.reload.get(self.api).format("x", "y")
            else:
                x = self.data["x"]
                expression = self.reload.get(self.api).format("x")
            res = eval(expression)
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def func(x, y):
                res = eval(expression)
                res.backward(grad_tensor)

            def func_x(x):
                res = eval(expression)
                res.backward(grad_tensor)

            for i in range(self.loops):
                if "y" in self.data.keys():
                    total_time = timeit.timeit(lambda: func(x, y), number=self.base_times)
                else:
                    total_time = timeit.timeit(lambda: func_x(x), number=self.base_times)
                self.total_time.append(total_time)
        else:
            raise AttributeError

    def run(self):
        """
        主执行函数，本地调试用
        """
        # 前反向时间
        self._run_forward()
        if self.enable_backward:
            self._run_total()
        # # 数据处理
        self._compute()
        # # 数据对比打印
        self._show()

    def run_schedule(self):
        """
        例行执行，会写文件
        """
        # 前反向时间
        self._run_forward()
        if self.enable_backward:
            self._run_total()
        # 数据处理
        self._compute()
        # 写文件
        self._save(self.result)

    def _run_forward(self):
        """
        测试前向时间
        """
        if self.framework == "paddle":
            self.paddle_forward()

    def _run_total(self):
        """
        测试总时间
        """
        if self.framework == "paddle":
            self.paddle_total()

    def _return_forward(self):
        """
        返回前向时间
        """
        # 前反向时间
        self._run_forward()
        # if self.enable_backward:
        #     self._run_total()
        # 数据处理
        head = int(self.loops / 5)
        tail = int(self.loops - self.loops / 5)
        res = sum(sorted(self.forward_time)[head:tail]) / (tail - head)
        return res

    def _compute(self):
        """
        数据处理
        """
        head = int(self.loops / 5)
        tail = int(self.loops - self.loops / 5)
        self.result["forward"] = ACCURACY % (sum(sorted(self.forward_time)[head:tail]) / (tail - head))
        if self.enable_backward:
            self.result["total"] = ACCURACY % (sum(sorted(self.total_time)[head:tail]) / (tail - head))
            self.result["backward"] = ACCURACY % (float(self.result["total"]) - float(self.result["forward"]))
            self.result["best_total"] = ACCURACY % min(self.total_time)
        else:
            self.result["total"] = self.result["forward"]
            self.result["backward"] = 0
            self.result["best_total"] = ACCURACY % min(self.forward_time)

    def _show(self):
        """
        logger 打印
        """
        self.logger.info("{} {} times forward cost {}s".format(self.framework, self.base_times, self.result["forward"]))
        self.logger.info(
            "{} {} times backward cost {}s".format(self.framework, self.base_times, self.result["backward"])
        )
        self.logger.info("{} {} times total cost {}s".format(self.framework, self.base_times, self.result["total"]))
        self.logger.info(
            "{} {} times best_total cost {}s".format(self.framework, self.base_times, self.result["best_total"])
        )

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
