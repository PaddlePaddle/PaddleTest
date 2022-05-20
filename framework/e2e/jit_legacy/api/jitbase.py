#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jit framework
"""
import os
import shutil
import logging
import pytest
import paddle
import numpy as np


seed = 33
np.random.seed(seed)
paddle.seed(seed)


class Runner(object):
    """
    Runner
    """

    def __init__(self, func, name, dtype=["float64"], ftype="func"):
        """
        init
        """
        pwd = os.getcwd()
        self.root_path = os.path.join(pwd, "save")
        self.func = func
        self.name = name
        self.delta = 1e-10
        self.rtol = 1e-11
        self.debug = True
        self.places = ["cpu", "gpu:0"]
        self.dtype = dtype
        self.ftype = ftype
        if paddle.device.is_compiled_with_cuda() is not True:
            self.places = ["cpu"]
        # 传参工具
        self.kwargs_dict = {"params_group1": {}}

    def add_kwargs_to_dict(self, group_name, **kwargs):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = kwargs

    def mkdir(self):
        """
        make save path
        """
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path)
        os.mkdir(self.root_path)

    def _mk_expect(self, func=None, dtype="float64"):
        """
        make expect output value
        """
        for k, v in self.kwargs_dict["params_group1"].items():
            if isinstance(v, (np.generic, np.ndarray)):
                self.kwargs_dict_exp["params_group1"][k] = paddle.to_tensor(v, dtype=dtype)

        exp = func(**self.kwargs_dict_exp["params_group1"])
        return exp

    def _jit_save(self, func=None):
        """
        jit save
        """
        self.save_path = os.path.join(self.root_path, self.name)
        paddle.jit.save(func, self.save_path)

    def _jit_load(self):
        """
        jit load
        """
        self.load_func = paddle.jit.load(self.save_path)

    def _mk_result(self, dtype="float64"):
        """
        make result output value
        """
        for i, v in enumerate(self.kwargs_dict["params_group1"].values()):
            if isinstance(v, (np.generic, np.ndarray)):
                self.kwargs_dict_res["params_group1"].append(v.astype(dtype))

        res = self.load_func(*self.kwargs_dict_res["params_group1"])
        return res

    def run(self):
        """
        main run
        """
        for place in self.places:
            self.place = place
            logging.info("[Place] is ===============================>>>>>>>>" + str(self.place))
            paddle.set_device(self.place)
            for _dtype in self.dtype:
                logging.info("[dtype] is +++++++++++++++++++++++++++++>>>>>>>>" + str(_dtype))
                self.kwargs_dict_exp = {"params_group1": {}}
                self.kwargs_dict_res = {"params_group1": []}
                if self.ftype == "layer":
                    func = self.func(_dtype)
                else:
                    func = self.func
                self.expect = self._mk_expect(func=func, dtype=_dtype)
                self._jit_save(func=func)
                logging.info("jit save [{}] has been done at [{}]".format(_dtype, str(self.place)))
                self._jit_load()
                logging.info("jit load [{}] has been done at [{}]".format(_dtype, str(self.place)))
                self.result = self._mk_result(dtype=_dtype)
                logging.info("make [{}] result at [{}]".format(_dtype, str(self.place)))
                self.compare(self.result.numpy(), self.expect.numpy(), self.delta, self.rtol)
                logging.info("compare [{}] result and expect at [{}]".format(_dtype, str(self.place)))

    def compare(self, result, expect, delta=1e-10, rtol=1e-11):
        """
        compare method
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
                    self.compare(j, expect[i], delta, rtol)
                else:
                    self.compare(j.numpy(), expect[i], delta, rtol)
        elif isinstance(result, str):
            res = result == expect
            if res is False:
                logging.error("the result is {}".format(result))
                logging.error("the expect is {}".format(expect))
            assert res
        else:
            assert result == pytest.approx(expect, delta)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)
