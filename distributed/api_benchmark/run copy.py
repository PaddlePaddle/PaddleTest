#!/bin/env python
# -*- coding: utf-8 -*-
import random
import timeit
import os
import json
from inspect import isclass
import paddle
import numpy as np
from paddle import to_tensor
from utils.logger import Logger
from reload_config import OPERATOR_RELOAD


# 计算精度，保留6位有效数字
ACCURACY = "%.6g"


class API_Benchmark(object):
    """
    compare tools
    """

    def __init__(
        self,
        api,
        logger,
        default_dtype="float32",
        place=None,
        card=None,
        title=None,
        param=None,
        loops=50,
        base_times=1000,
    ):
        """

        :param api: paddle的api
        :param place:  cpu or gpu (string)
        :param card: "0,1,2,3,4,5,6,7" (string)
        :param explain: case的说明 会打印在日志中
        """
        
        self.debug = True
        paddle.set_default_dtype(default_dtype)

        # 循环次数
        self.loops = loops
        # timeit 基础运行次数
        self.base_times = base_times
        # 设置logger
        self.logger = logger.get_log()


        self.result = {}

        # set api name
        self.result["api"] = api
        # set log file name
        self.log_file_name = title
        self.result["script"] = self.log_file_name
        # set Reload API DICT
        # self.reload = OPERATOR_RELOAD
        
        self.param = param
        # self.data = dict()
        # self.method = dict()
        self.places = place
        self.card = card
        self._set_place(self.card)


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

    def paddle_communicator(self):
        """
        主体测试逻辑:通信类API
        """
        cost_time_list = []
        algbw_list = []
        for i in range(self.loops):
            
            cmd = "python -m paddle.distributed.launch --devices=0,1,2,3,4,5,6,7 " + self.title + self.param
            pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = pro.communicate()
            print(out)
            
        
            

            def func(x, y):
                eval(expression)

            def func_x(x):
                eval(expression)

            for i in range(self.loops):
                if "y" in self.data.keys():
                    forward_time = timeit.timeit(lambda: func(x, y), number=self.base_times)
                else:
                    forward_time = timeit.timeit(lambda: func_x(x), number=self.base_times)
                forward_time_list.append(forward_time)
        else:
            raise AttributeError

        return forward_time_list

    def paddle_total(self):
        """
        计算paddle 总体时间
        """
        total_time_list = []
        if self._layertypes(self.api) == "func":
            input_param = dict(self.data, **self.param)
            res = self.api(**input_param)
            grad_tensor = paddle.ones(res.shape, res.dtype)

            def func(input_param):
                res = self.api(**input_param)
                res.backward(grad_tensor)

            for i in range(self.loops):
                total_time = timeit.timeit(lambda: func(input_param), number=self.base_times)
                total_time_list.append(total_time)
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
                total_time_list.append(total_time)
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
                total_time_list.append(total_time)
        else:
            raise AttributeError

        return total_time_list

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
    
    
    def _run_total(self):
        """
        测试总时间
        """
        if self.framework == "paddle":
            self.paddle_total()
    
    def _show(self, forward_time, backward_time, total_time, best_total_time):
        """
        logger 打印
        """
        self.logger.info("{} {} times forward cost {}s".format(self.framework, self.base_times, forward_time))
        self.logger.info("{} {} times backward cost {}s".format(self.framework, self.base_times, backward_time))
        self.logger.info("{} {} times total cost {}s".format(self.framework, self.base_times, total_time))
        self.logger.info("{} {} times best_total cost {}s".format(self.framework, self.base_times, best_total_time))

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
