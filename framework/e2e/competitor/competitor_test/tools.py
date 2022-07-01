#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tools
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "utils"))

import numpy as np
import paddle
import torch
import pytest
from utils.logger import Logger


class FrontAPIBase(object):
    """
    convert class to function
    """

    def __init__(self, func):
        """initialize"""
        self.api = func

    def encapsulation(self, *args, **kwargs):
        """class to func"""
        obj = self.api(**kwargs)
        return obj(*args)

    def exe(self, *args, **kwargs):
        """run"""
        return self.encapsulation(*args, **kwargs)


def compare(paddle, torch, delta=1e-6, rtol=1e-5):
    """
    比较函数
    :param paddle: paddle结果
    :param torch: torch结果
    :param delta: 误差值
    :return:
    """
    logger = Logger("compare")
    if isinstance(paddle, np.ndarray):
        expect = np.array(torch)
        res = np.allclose(paddle, torch, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            logger.get_log().error("the paddle is {}".format(paddle))
            logger.get_log().error("the torch is {}".format(torch))
        # tools.assert_true(res)
        assert res
        # tools.assert_equal(result.shape, expect.shape)
        if len(paddle.shape) != 1:
            assert paddle.shape == expect.shape
    elif isinstance(paddle, (list, tuple)):
        for i, j in enumerate(paddle):
            if isinstance(j, (np.generic, np.ndarray)):
                compare(j, torch[i], delta, rtol)
            else:
                compare(j.numpy(), torch[i], delta, rtol)
    elif isinstance(paddle, str):
        res = paddle == torch
        if res is False:
            logger.get_log().error("the paddle is {}".format(paddle))
            logger.get_log().error("the torch is {}".format(torch))
        assert res
    else:
        assert paddle == pytest.approx(torch, delta)


def solve_tuple(item, type, convert):
    """
    solve tuple
    """
    if isinstance(item, type):
        return convert(item)
    elif isinstance(item, (list, tuple)):
        item = list(item)
        length = len(item)
        for j in range(length):
            item[j] = solve_tuple(item[j], type, convert)
        return item
    else:
        return item


TORCHDTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bool": torch.bool,
}


TORCHDEVICE = {"cpu": torch.device("cpu"), "gpu": torch.device("cuda"), "gpu:0": torch.device("cuda:0")}


COMPAREGAP = {"UpsamplingBilinear2D": 1e-4, "selu": 1e-4}


def case2name(api_name, n):
    """
    api_name convert api cases
    """
    s = ["_base"] + ["_" + str(k) for k in range(n + 1)]
    x = [api_name + i for i in s]
    return x


STOP_BACKWARD = [
    "add_1",
    *case2name("allclose", 9),
    *case2name("equal_all", 4),
    *case2name("greater_equal", 2),
    *case2name("less_equal", 2),
    *case2name("tanh_", 2),
]
