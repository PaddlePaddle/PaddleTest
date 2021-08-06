#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_logsoftmax
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestLogSoftmax(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestLogSoftmax(paddle.nn.LogSoftmax)


@pytest.mark.api_nn_LOGSOFTMAX_vartype
def test_logsoftmax_base():
    """
    base
    """
    x = np.array([[[1], [2]], [[3], [4]]]).astype(np.float32)
    res = x.reshape(2, 2)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(2, 2, 1))
    obj.base(res=res, data=x, axis=0)


@pytest.mark.api_nn_LOGSOFTMAX_parameters
def test_logsoftmax():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    res = x.transpose(2, 0, 1).reshape(4, 6)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(4, 2, 3).transpose(1, 2, 0))
    obj.run(res=res, data=x)


@pytest.mark.api_nn_LOGSOFTMAX_parameters
def test_logsoftmax1():
    """
    axis=-1
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    # 算法
    res = x.transpose(2, 0, 1).reshape(4, 6)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(4, 2, 3).transpose(1, 2, 0))
    obj.run(res=res, data=x, axis=-1)


@pytest.mark.api_nn_LOGSOFTMAX_parameters
def test_logsoftmax2():
    """
    axis=0
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    res = x.reshape(2, 12)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(2, 3, 4))
    obj.run(res=res, data=x, axis=0)


@pytest.mark.api_nn_LOGSOFTMAX_parameters
def test_logsoftmax3():
    """
    axis=1
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    # 算法
    res = x.transpose(1, 0, 2).reshape(3, 8)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(3, 2, 4).transpose(1, 0, 2))
    obj.run(res=res, data=x, axis=1)


@pytest.mark.api_nn_LOGSOFTMAX_parameters
def test_logsoftmax4():
    """
    axis=2
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    # 算法
    res = x.transpose(2, 0, 1).reshape(4, 6)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(4, 2, 3).transpose(1, 2, 0))
    obj.run(res=res, data=x, axis=2)


@pytest.mark.api_nn_LOGSOFMAX_exception
def test_logsoftmax5():
    """
    axis=3
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    # 算法
    res = x.transpose(2, 0, 1).reshape(4, 6)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(4, 2, 3).transpose(1, 2, 0))
    obj.exception(etype="InvalidArgumentError", data=x, axis=3)


@pytest.mark.api_nn_LOGSOFTMAX_exception
def test_logsoftmax6():
    """
    axis="3"
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    # 算法
    res = x.transpose(2, 0, 1).reshape(4, 6)
    res = np.log((np.exp(res) / sum(np.exp(res))).reshape(4, 2, 3).transpose(1, 2, 0))
    obj.exception(etype="InvalidArgumentError", data=x, axis="3")
