#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jelly 主执行逻辑文件
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


class Jelly(object):
    """
    compare tools
    """

    def __init__(self, paddle_api, torch_api, default_dtype="float32", place=None, card=None, title=None):
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

        paddle.set_default_dtype(PADDLE_DTYPE[default_dtype])
        torch.set_default_dtype(TORCH_DTYPE[default_dtype])

        # 循环次数
        self.loops = 50
        # timeit 基础运行时间
        self.base_times = 1000
        # 设置logger
        self.logger = logger.get_log()

        self.paddle_forward_time = []
        self.paddle_backward_time = []
        self.paddle_total_time = []
        self.paddle_loss_time = []
        self.torch_forward_time = []
        self.torch_backward_time = []
        self.torch_loss_time = []
        self.torch_total_time = []
        self.paddle_total_time = []
        self.torch_total_time = []

        self.dump_data = []
        self.result = {"paddle": dict(), "torch": dict(), "compare": dict()}

        # set api name
        self.result["paddle"]["api"] = paddle_api
        self.result["torch"]["api"] = torch_api
        # set log file name
        self.log_file_name = title
        self.result["yaml"] = self.log_file_name
        # trans "str api" to obj
        self.paddle_api = eval(paddle_api)
        self.torch_api = eval(torch_api)
        self.compare_dict = None
        self.paddle_param = dict()
        self.paddle_data = dict()
        self.torch_param = dict()
        self.torch_data = dict()
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
            self.paddle_data[key] = to_tensor(value)
            self.paddle_data[key].stop_gradient = False
        for key, value in param.items():
            if isinstance(value, (np.generic, np.ndarray)):
                self.paddle_param[key] = to_tensor(value)
            else:
                self.paddle_param[key] = value

    def set_torch_param(self, inputs: dict, param: dict):
        """
        设置torch 输入参数
        """
        for key, value in inputs.items():
            if self.places == "cpu":
                self.torch_data[key] = torch.tensor(value)
            else:
                self.torch_data[key] = torch.tensor(value).to("cuda")
            self.torch_data[key].requires_grad = True
        for key, value in param.items():
            if isinstance(value, (np.generic, np.ndarray)):
                if self.places == "cpu":
                    self.torch_param[key] = torch.tensor(value)
                else:
                    self.torch_param[key] = torch.tensor(value).to("cuda")
            else:
                self.torch_param[key] = value

    def paddle_forward(self):
        """
        主体测试逻辑
        """
        if self._layertypes(self.paddle_api) == "func":
            input_param = dict(self.paddle_data, **self.paddle_param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: self.paddle_api(**input_param), number=self.base_times)
                self.paddle_forward_time.append(forward_time)
        elif self._layertypes(self.paddle_api) == "class":
            obj = self.paddle_api(**self.paddle_param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: obj(**self.paddle_data), number=self.base_times)
                self.paddle_forward_time.append(forward_time)
        else:
            raise AttributeError

    def paddle_total(self):
        """
        计算paddle 总体时间
        """
        if self._layertypes(self.paddle_api) == "func":
            input_param = dict(self.paddle_data, **self.paddle_param)
            res = self.paddle_api(**input_param)
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def func(input_param):
                res = self.paddle_api(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: func(input_param), number=self.base_times)
                self.paddle_total_time.append(total_time)
        elif self._layertypes(self.paddle_api) == "class":
            obj = self.paddle_api(**self.paddle_param)
            res = obj(**self.paddle_data)
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def clas(input_param):
                res = obj(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: clas(self.paddle_data), number=self.base_times)
                self.paddle_total_time.append(total_time)
        else:
            raise AttributeError

    def torch_forward(self):
        """
        torch 前向时间
        """
        if self._layertypes(self.torch_api) == "func":
            input_param = dict(self.torch_data, **self.torch_param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: self.torch_api(**input_param), number=self.base_times)
                self.torch_forward_time.append(forward_time)
        elif self._layertypes(self.torch_api) == "class":
            obj = self.torch_api(**self.torch_param)
            for i in range(self.loops):
                forward_time = timeit.timeit(lambda: obj(**self.torch_data), number=self.base_times)
                self.torch_forward_time.append(forward_time)
        else:
            raise AttributeError

    def torch_total(self):
        """
        torch 总时间
        """
        if self._layertypes(self.torch_api) == "func":
            input_param = dict(self.torch_data, **self.torch_param)
            res = self.torch_api(**input_param)
            # init grad tensor
            if self.places == "gpu":
                grad_tensor = torch.ones(res.shape, dtype=res.dtype).to("cuda")
            else:
                grad_tensor = torch.ones(res.shape, dtype=res.dtype)

            def func(input_param):
                res = self.torch_api(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: func(input_param), number=self.base_times)
                self.torch_total_time.append(total_time)
        elif self._layertypes(self.torch_api) == "class":
            obj = self.torch_api(**self.torch_param)
            res = obj(**self.torch_data)
            # init grad tensor
            grad_tensor = torch.ones(res.shape, dtype=res.dtype)

            def clas(input_param):
                """ lambda clas"""
                res = obj(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: clas(self.torch_data), number=self.base_times)
                self.torch_total_time.append(total_time)
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
        self.paddle_forward()
        self.torch_forward()

    def _run_total(self):
        """
        测试总时间
        """
        self.torch_total()
        self.paddle_total()

    def _compute(self):
        """
        数据处理
        """
        head = int(self.loops / 5)
        tail = int(self.loops - self.loops / 5)
        self.result["paddle"]["forward"] = sum(sorted(self.paddle_forward_time)[head:tail]) / (tail - head)
        self.result["paddle"]["total"] = sum(sorted(self.paddle_total_time)[head:tail]) / (tail - head)
        self.result["paddle"]["backward"] = self.result["paddle"]["total"] - self.result["paddle"]["forward"]
        self.result["torch"]["forward"] = sum(sorted(self.torch_forward_time)[head:tail]) / (tail - head)
        self.result["torch"]["total"] = sum(sorted(self.torch_total_time)[head:tail]) / (tail - head)
        self.result["torch"]["backward"] = self.result["torch"]["total"] - self.result["torch"]["forward"]

        # compare
        if self.result["paddle"]["forward"] > self.result["torch"]["forward"]:
            forward = (self.result["paddle"]["forward"] / self.result["torch"]["forward"]) * -1
        else:
            forward = self.result["torch"]["forward"] / self.result["paddle"]["forward"]

        if self.result["paddle"]["backward"] > self.result["torch"]["backward"]:
            backward = (self.result["paddle"]["backward"] / self.result["torch"]["backward"]) * -1
        else:
            backward = self.result["torch"]["backward"] / self.result["paddle"]["backward"]

        if self.result["paddle"]["total"] > self.result["torch"]["total"]:
            total = (self.result["paddle"]["total"] / self.result["torch"]["total"]) * -1
        else:
            total = self.result["torch"]["total"] / self.result["paddle"]["total"]

        self.result["compare"]["forward"] = forward
        self.result["compare"]["backward"] = backward
        self.result["compare"]["total"] = total
        # print(self.result)

    def _show(self):
        """
        logger 打印
        """
        self.logger.info(
            "paddle {} times forward cost {:.5f}s".format(self.base_times, self.result["paddle"]["forward"])
        )
        self.logger.info(
            "paddle {} times backward cost {:.5f}s".format(self.base_times, self.result["paddle"]["backward"])
        )
        self.logger.info("paddle {} times total cost {:.5f}s".format(self.base_times, self.result["paddle"]["total"]))
        self.logger.info("torch {} times forward cost {:.5f}s".format(self.base_times, self.result["torch"]["forward"]))
        self.logger.info(
            "torch {} times backward cost {:.5f}s".format(self.base_times, self.result["torch"]["backward"])
        )
        self.logger.info("torch {} times total cost {:.5f}s".format(self.base_times, self.result["torch"]["total"]))

        if self.result["paddle"]["forward"] > self.result["torch"]["forward"]:
            forward = "forward: torch is {:.3f}x faster than paddle".format(
                self.result["paddle"]["forward"] / self.result["torch"]["forward"]
            )
        else:
            forward = "forward: paddle is {:.3f}x faster than torch".format(
                self.result["torch"]["forward"] / self.result["paddle"]["forward"]
            )
        self.logger.info(forward)

        if self.result["paddle"]["backward"] > self.result["torch"]["backward"]:
            backward = "backward: torch is {:.3f}x faster than paddle".format(
                self.result["paddle"]["backward"] / self.result["torch"]["backward"]
            )
        else:
            backward = "backward: paddle is {:.3f}x faster than torch".format(
                self.result["torch"]["backward"] / self.result["paddle"]["backward"]
            )
        self.logger.info(backward)

        if self.result["paddle"]["total"] > self.result["torch"]["total"]:
            total = "Total: torch is {:.3f}x faster than paddle".format(
                self.result["paddle"]["total"] / self.result["torch"]["total"]
            )
        else:
            total = "Total: paddle is {:.3f}x faster than torch".format(
                self.result["torch"]["total"] / self.result["paddle"]["total"]
            )
        self.logger.info(total)

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
