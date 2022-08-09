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
import torch
import numpy as np
from paddle import to_tensor
from utils.logger import logger


PADDLE_DTYPE = {"float32": np.float32, "float64": np.float64}
TORCH_DTYPE = {"float32": torch.float32, "float64": torch.float64}
# 计算精度，保留6位有效数字
ACCURACY = "%.6g"


class Jelly_v2(object):
    """
    compare tools
    """

    def __init__(self, api, framework="paddle", default_dtype="float32", place=None, card=None, title=None):
        """

        :param paddle_api:
        :param torch_api:
        :param place:  cpu or gpu (string)
        :param card: 0 1 2 3 (int)
        :param explain: case的说明 会打印在日志中
        """
        self.seed = 33
        self.enable_backward = True
        self.debug = True
        self.framework = framework

        paddle.set_default_dtype(PADDLE_DTYPE[default_dtype])
        torch.set_default_dtype(TORCH_DTYPE[default_dtype])

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
        # trans "str api" to obj
        self.api = eval(api)
        self.compare_dict = None
        self.param = dict()
        self.data = dict()
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
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        use_cuda = paddle.is_compiled_with_cuda()
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

    def _set_place(self, card=None):
        """
        init place
        :return:
        """
        if self.places is None:
            if paddle.is_compiled_with_cuda() is True:
                self.places = "gpu"
                if torch.cuda.is_available() is True:
                    if card is None:
                        paddle.set_device("gpu:0")
                        torch.device(0)
                    else:
                        paddle.set_device("gpu:{}".format(card))
                        torch.device(card)
                else:
                    raise EnvironmentError
            else:
                self.places = "cpu"
                paddle.set_device("cpu")
                torch.device("cpu")
        else:
            if self.places == "cpu":
                paddle.set_device("cpu")
                torch.device("cpu")
            else:
                if card is None:
                    paddle.set_device("gpu:0")
                    torch.device(0)
                else:
                    paddle.set_device("gpu:{}".format(card))
                    torch.device(card)

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

    # def baseline(self):
    #     print(timeit.timeit('"-".join(str(n) for n in range(100))', number=10000))

    def set_paddle_param(self, inputs: dict, param: dict):
        """
        设置paddle 输入参数
        """
        for key, value in inputs.items():
            self.data[key] = to_tensor(value)
            self.data[key].stop_gradient = False
        for key, value in param.items():
            if isinstance(value, (np.generic, np.ndarray)):
                self.param[key] = to_tensor(value)
            else:
                self.param[key] = value

    def set_torch_param(self, inputs: dict, param: dict):
        """
        设置torch 输入参数
        """
        for key, value in inputs.items():
            if self.places == "cpu":
                self.data[key] = torch.tensor(value)
            else:
                self.data[key] = torch.tensor(value).to("cuda")
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

    def paddle_forward(self):
        """
        主体测试逻辑
        """
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: self.api(**input_param), number=self.base_times)
                self.forward_time.append(forward_time)
        elif self._layertypes(self.api) == "class":
            obj = self.api(**self.param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: obj(*self.data.values()), number=self.base_times)
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
            obj = self.api(**self.param)
            res = obj(*self.data.values())
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def clas(input_param):
                res = obj(*input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: clas(self.data.values()), number=self.base_times)
                self.total_time.append(total_time)
        else:
            raise AttributeError

    def torch_forward(self):
        """
        torch 前向时间
        """
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: self.api(**input_param), number=self.base_times)
                self.forward_time.append(forward_time)
        elif self._layertypes(self.api) == "class":
            obj = self.api(**self.param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: obj(*self.data.values()), number=self.base_times)
                self.forward_time.append(forward_time)
        else:
            raise AttributeError

    def torch_total(self):
        """
        torch 总时间
        """
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

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: func(input_param), number=self.base_times)
                self.total_time.append(total_time)
        elif self._layertypes(self.api) == "class":
            obj = self.api(**self.param)
            res = obj(*self.data.values())
            # init grad tensor
            if self.places == "gpu":
                grad_tensor = torch.ones(res.shape, dtype=res.dtype).to("cuda")
            else:
                grad_tensor = torch.ones(res.shape, dtype=res.dtype)

            def clas(input_param):
                """ lambda clas"""
                res = obj(*input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: clas(self.data.values()), number=self.base_times)
                self.total_time.append(total_time)
        else:
            raise AttributeError

    def run(self):
        """
        主执行函数，本地调试用
        """
        # 前反向时间
        self._run_forward()
        self._run_total()
        # 数据处理
        self._compute()
        # 数据对比打印
        self._show()

    def run_schedule(self):
        """
        例行执行，会写文件
        """
        # 前反向时间
        self._run_forward()
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
        elif self.framework == "torch":
            self.torch_forward()

    def _run_total(self):
        """
        测试总时间
        """
        if self.framework == "paddle":
            self.paddle_total()
        elif self.framework == "torch":
            self.torch_total()

    def _compute(self):
        """
        数据处理
        """
        head = int(self.loops / 5)
        tail = int(self.loops - self.loops / 5)
        self.result["forward"] = ACCURACY % (sum(sorted(self.forward_time)[head:tail]) / (tail - head))
        self.result["total"] = ACCURACY % (sum(sorted(self.total_time)[head:tail]) / (tail - head))
        self.result["backward"] = ACCURACY % (float(self.result["total"]) - float(self.result["forward"]))

    def _show(self):
        """
        logger 打印
        """
        self.logger.info("{} {} times forward cost {}s".format(self.framework, self.base_times, self.result["forward"]))
        self.logger.info(
            "{} {} times backward cost {}s".format(self.framework, self.base_times, self.result["backward"])
        )
        self.logger.info("{} {} times total cost {}s".format(self.framework, self.base_times, self.result["total"]))

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
