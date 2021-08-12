#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
loss case runner
"""
from inspect import isfunction
import paddle
import numpy as np

seed = 33
np.random.seed(seed)
paddle.seed(seed)


class Runner(object):
    """Runner"""

    def __init__(self, reader, label, model, loss, **args):
        """init"""
        self.reader = paddle.to_tensor(reader)
        self.label = paddle.to_tensor(label)
        self.model = model
        self.learning_rate = 0.001
        self.optimizer = paddle.optimizer.SGD(self.learning_rate, parameters=self.model.parameters())
        self.loss = loss
        self.debug = False
        self.softmax = False
        self.result = []
        types = {0: "func", 1: "class"}
        # 设置函数执行方式，函数式还是声明式.
        if isfunction(self.loss):
            self.__layertype = types[0]
        else:
            self.__layertype = types[1]
        # 传参工具
        self.kwargs_dict = {"params_group1": {}, "params_group2": {}}

    def add_kwargs_to_dict(self, group_name, **kwargs):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = kwargs

    def run(self):
        """run your models"""
        if self.__layertype == "func":
            for i in range(10):
                out = self.model(self.reader)
                if self.softmax is True:
                    out = paddle.nn.functional.softmax(out)
                loss = self.loss(out, self.label, **self.kwargs_dict["params_group1"])
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                if self.debug:
                    print(loss)
                self.result.append(loss.numpy()[0])
        elif self.__layertype == "class":
            for i in range(10):
                out = self.model(self.reader)
                if self.softmax is True:
                    out = paddle.nn.functional.softmax(out)
                obj = self.loss(**self.kwargs_dict["params_group1"])
                loss = obj(out, self.label, **self.kwargs_dict["params_group2"])
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                if self.debug:
                    print(loss)
                self.result.append(loss.numpy()[0])

    def check(self, expect=None):
        """
        check result
        """
        if self.result is None:
            raise Exception("Model result is None， check your code")
        if self.debug:
            print(self.result)
        try:
            assert np.allclose(self.result, expect), "Error in check loss"
        except Exception as e:
            print(e)
            print("expect loss is {}".format(expect))
            print("Model result is {}".format(self.result))
            assert False
