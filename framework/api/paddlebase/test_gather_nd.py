#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_gather_nd
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestGather(APIBase):
    """
    test gather_nd
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        self.enable_backward = False
        self.no_grad_var = ["index", "axis"]


obj = TestGather(paddle.gather_nd)


def cal_gather_nd(x, index):
    """
    calculate gather_nd
    """
    data = x.tolist()
    print(data)
    idx = index.flatten().tolist()
    for i in idx:
        data = data[i]
    res = np.array(data)
    length = len(index.shape)
    if length > 1:
        for i in range(length - 1):
            res = res[np.newaxis, :]
    return res


@pytest.mark.api_base_gather_nd_vartype
def test_gather_nd_base():
    """
    base
    index is int32
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2]).astype("int32")
    res = cal_gather_nd(x, index)
    obj.base(res=res, x=x, index=index)


@pytest.mark.api_base_gather_nd_vartype
def test_gather_nd0():
    """
    base
    index is int64
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2]).astype("int64")
    res = cal_gather_nd(x, index)
    obj.base(res=res, x=x, index=index)


@pytest.mark.api_base_gather_nd_parameters
def test_gather_nd1():
    """
    x: multidimensional
    index is int64
    """
    x = np.arange(12).reshape((1, 2, 3, 2))
    index = np.array([0, 1, 2]).astype("int64")
    res = cal_gather_nd(x, index)
    obj.run(res=res, x=x, index=index)


@pytest.mark.api_base_gather_nd_parameters
def test_gather_nd2():
    """
    x: multidimensional
    index: multidimensional
    """
    x = np.arange(12).reshape((1, 2, 3, 2))
    index = np.array([[[[0, 1, 2]]]]).astype("int64")
    res = cal_gather_nd(x, index)
    obj.run(res=res, x=x, index=index)


@pytest.mark.api_base_gather_nd_exception
def test_gather_nd3():
    """
    index: out of range
    """
    x = np.arange(12).reshape((2, 3, 2))
    index = np.array([[[[0, 1, 2, 4]]]]).astype("int64")
    obj.exception(ValueError, mode="python", x=x, index=index)
