#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
loss case runner
"""
from inspect import isfunction
import logging
import copy
import pytest
import paddle
import numpy as np


seed = 33
np.random.seed(seed)
paddle.seed(seed)


class Runner(object):
    """Runner"""

    def __init__(self, reader, loss):
        """init"""
        self.reader = reader
        self.learning_rate = 0.001
        self.loss = loss
        self.debug = False
        self.softmax = False
        self.dygraph = True
        self.static = True
        # self.case_name = '_base'
        self.np_type = None
        types = {0: "func", 1: "class"}
        # 设置函数执行方式，函数式还是声明式.
        if isfunction(self.loss):
            self.__layertype = types[0]
        else:
            self.__layertype = types[1]
        # 传参工具
        self.kwargs_dict = {"params_group1": {}, "params_group2": {}, "params_group3": {}}

    def add_kwargs_to_dict(self, group_name, **kwargs):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = kwargs

    def run(self, model=None, expect=None):
        """run your models"""
        if self.__layertype == "func":
            # test dygraph
            if self.dygraph is True:
                paddle.disable_static()
                self.result = []
                self.reader_dygraph = copy.deepcopy(self.reader)
                self.reader_dygraph = paddle.to_tensor(self.reader_dygraph)
                self.kwargs_dict_dygraph = copy.deepcopy(self.kwargs_dict)
                self.model_dygraph = model(**self.kwargs_dict_dygraph["params_group3"])
                self.optimizer_dygraph = paddle.optimizer.SGD(
                    self.learning_rate, parameters=self.model_dygraph.parameters()
                )
                for k, v in self.kwargs_dict_dygraph["params_group1"].items():
                    if isinstance(v, (np.generic, np.ndarray)):
                        if self.np_type is not None and k in self.np_type:
                            self.kwargs_dict_dygraph["params_group1"][k] = v
                        else:
                            self.kwargs_dict_dygraph["params_group1"][k] = paddle.to_tensor(v)
                for i in range(10):
                    out = self.model_dygraph(self.reader_dygraph)
                    if self.softmax is True:
                        out = paddle.nn.functional.softmax(out)
                    loss = self.loss(out, **self.kwargs_dict_dygraph["params_group1"])
                    loss.backward()
                    self.optimizer_dygraph.step()
                    self.optimizer_dygraph.clear_grad()
                    # logging.info('at {}, res is: {}'.format(i, loss))
                    if self.debug:
                        print(loss)
                    self.result.append(loss.numpy()[0])
                # logging.info('at {}, result is: {}'.format(i, self.result))
                self.check(result=self.result, expect=expect)

            if self.static is True:
                paddle.enable_static()
                self.result = []
                self.reader_data = copy.deepcopy(self.reader)
                self.kwargs_dict_static = copy.deepcopy(self.kwargs_dict)
                self.model_static = model(**self.kwargs_dict_dygraph["params_group3"])
                self.optimizer_static = paddle.optimizer.SGD(
                    self.learning_rate, parameters=self.model_static.parameters()
                )
                main_program = paddle.static.default_main_program()
                startup_program = paddle.static.default_startup_program()
                with paddle.utils.unique_name.guard():
                    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                        self.reader_static = paddle.static.data(
                            name="reader_static", shape=self.reader_data.shape, dtype=self.reader_data.dtype
                        )
                        feed_dict = {"reader_static": self.reader_data}
                        for k, v in self.kwargs_dict_static["params_group1"].items():
                            if isinstance(v, (np.generic, np.ndarray)):
                                if self.np_type is not None and k in self.np_type:
                                    self.kwargs_dict_static["params_group1"][k] = v
                                else:
                                    self.kwargs_dict_static["params_group1"][k] = paddle.static.data(
                                        name=k, shape=v.shape, dtype=v.dtype
                                    )
                                    feed_dict[k] = v
                        # logging.info("feed_dict is: {}".format(feed_dict))
                        # logging.info("params_group1 is: {}".format(self.kwargs_dict_static["params_group1"]))
                        out = self.model_static(self.reader_static)
                        if self.softmax is True:
                            out = paddle.nn.functional.softmax(out)
                        loss = self.loss(out, **self.kwargs_dict_static["params_group1"])
                        # logging.info('loss is: {}'.format(loss))
                        # logging.info('loss.shape is: {}'.format(loss.shape))
                        self.optimizer_static.minimize(loss)
                        exe = paddle.static.Executor()
                        exe.run(startup_program)
                        for i in range(10):
                            res = exe.run(main_program, feed=feed_dict, fetch_list=[loss], return_numpy=True)
                            # logging.info('at {}, res is: {}'.format(i, res))
                            if self.debug:
                                print(res[0])
                            self.result.append(res[0])
                        # logging.info('at {}, result is: {}'.format(i, self.result))
                        self.check(result=self.result, expect=expect)

        elif self.__layertype == "class":
            # test dygraph
            if self.dygraph is True:
                paddle.disable_static()
                self.result = []
                self.reader_dygraph = copy.deepcopy(self.reader)
                self.reader_dygraph = paddle.to_tensor(self.reader_dygraph)
                self.kwargs_dict_dygraph = copy.deepcopy(self.kwargs_dict)
                self.model_dygraph = model(**self.kwargs_dict_dygraph["params_group3"])
                self.optimizer_dygraph = paddle.optimizer.SGD(
                    self.learning_rate, parameters=self.model_dygraph.parameters()
                )
                for k, v in self.kwargs_dict_dygraph["params_group1"].items():
                    if isinstance(v, (np.generic, np.ndarray)):
                        if self.np_type is not None and k in self.np_type:
                            self.kwargs_dict_dygraph["params_group1"][k] = v
                        else:
                            self.kwargs_dict_dygraph["params_group1"][k] = paddle.to_tensor(v)
                for k, v in self.kwargs_dict_dygraph["params_group2"].items():
                    if isinstance(v, (np.generic, np.ndarray)):
                        if self.np_type is not None and k in self.np_type:
                            self.kwargs_dict_dygraph["params_group2"][k] = v
                        else:
                            self.kwargs_dict_dygraph["params_group2"][k] = paddle.to_tensor(v)
                for i in range(10):
                    out = self.model_dygraph(self.reader_dygraph)
                    # logging.info('at {}, model out.shape is: {}'.format(i, out.shape))
                    if self.softmax is True:
                        out = paddle.nn.functional.softmax(out)
                    obj = self.loss(**self.kwargs_dict_dygraph["params_group1"])
                    loss = obj(out, **self.kwargs_dict_dygraph["params_group2"])
                    # logging.info('at {}, loss.shape is: {}'.format(i, loss.shape))
                    loss.backward()
                    self.optimizer_dygraph.step()
                    self.optimizer_dygraph.clear_grad()
                    # logging.info('at {}, res is: {}'.format(i, loss))
                    if self.debug:
                        print(loss)
                    self.result.append(loss.numpy()[0])
                # logging.info('at {}, result is: {}'.format(i, self.result))
                self.check(result=self.result, expect=expect)

            if self.static is True:
                paddle.enable_static()
                self.result = []
                self.reader_data = copy.deepcopy(self.reader)
                self.kwargs_dict_static = copy.deepcopy(self.kwargs_dict)
                self.model_static = model(**self.kwargs_dict_dygraph["params_group3"])
                self.optimizer_static = paddle.optimizer.SGD(
                    self.learning_rate, parameters=self.model_static.parameters()
                )
                main_program = paddle.static.default_main_program()
                startup_program = paddle.static.default_startup_program()
                with paddle.utils.unique_name.guard():
                    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                        self.reader_static = paddle.static.data(
                            name="reader_static", shape=self.reader_data.shape, dtype=self.reader_data.dtype
                        )
                        feed_dict = {"reader_static": self.reader_data}
                        for k, v in self.kwargs_dict_static["params_group1"].items():
                            if isinstance(v, (np.generic, np.ndarray)):
                                if self.np_type is not None and k in self.np_type:
                                    self.kwargs_dict_static["params_group1"][k] = v
                                else:
                                    self.kwargs_dict_static["params_group1"][k] = paddle.static.data(
                                        name=k, shape=v.shape, dtype=v.dtype
                                    )
                                    feed_dict[k] = v
                        for k, v in self.kwargs_dict_static["params_group2"].items():
                            if isinstance(v, (np.generic, np.ndarray)):
                                if self.np_type is not None and k in self.np_type:
                                    self.kwargs_dict_static["params_group2"][k] = v
                                else:
                                    self.kwargs_dict_static["params_group2"][k] = paddle.static.data(
                                        name=k, shape=v.shape, dtype=v.dtype
                                    )
                                    feed_dict[k] = v
                        # logging.info("feed_dict is: {}".format(feed_dict))
                        # logging.info("params_group2 is: {}".format(self.kwargs_dict_static["params_group2"]))
                        out = self.model_static(self.reader_static)
                        if self.softmax is True:
                            out = paddle.nn.functional.softmax(out)
                        obj = self.loss(**self.kwargs_dict_static["params_group1"])
                        print(self.kwargs_dict_static["params_group2"])
                        loss = obj(out, **self.kwargs_dict_static["params_group2"])
                        # logging.info('loss is: {}'.format(loss))
                        self.optimizer_static.minimize(loss)
                        exe = paddle.static.Executor()
                        exe.run(startup_program)
                        for i in range(10):
                            res = exe.run(main_program, feed=feed_dict, fetch_list=[loss], return_numpy=True)
                            # logging.info('at {}, res is: {}'.format(i, res))
                            if self.debug:
                                print(res[0])
                            self.result.append(res[0])
                        # logging.info('at {}, result is: {}'.format(i, self.result))
                        self.check(result=self.result, expect=expect)

    def check(self, result=None, expect=None):
        """
        check result
        """
        if result is None:
            raise Exception("Model result is None， check your code")
        for i, v in enumerate(result):
            if isinstance(v, np.ndarray):
                result[i] = result[i][0]
        if self.debug:
            print(result)
        try:
            assert np.allclose(result, expect), "Error in check loss"
        except Exception as e:
            print(e)
            print("expect loss is {}".format(expect))
            print("Model result is {}".format(result))
            assert False


def compare(result, expect, delta=1e-6, rtol=1e-7):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if isinstance(result, np.ndarray):
        expect = np.array(expect)
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        assert res
        assert result.shape == expect.shape
    elif isinstance(result, list):
        for i, j in enumerate(result):
            if isinstance(j, (np.generic, np.ndarray)):
                compare(j, expect[i], delta, rtol)
            else:
                compare(j.numpy(), expect[i], delta, rtol)
    elif isinstance(result, str):
        res = result == expect
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        assert res
    else:
        assert result == pytest.approx(expect, delta)
