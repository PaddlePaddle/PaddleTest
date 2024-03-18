#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_unsqueeze_
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestUnsqueeze(APIBase):
    """
    test unsqueeze_
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int8, np.int32, np.int64]
        self.no_grad_var = ["axis"]
        self.enable_backward = False
        # self.static = True
        # self.dygraph = True
        # self.debug = True


obj = TestUnsqueeze(paddle.unsqueeze_)


@pytest.mark.api_base_unsqueeze__vartype
def test_unsqueeze__base():
    """
    base
    axis=int32
    Cannot support x as bool tensor, it's a bug
    """
    x = np.arange(6).reshape(2, 3)
    axis = 1
    res = np.expand_dims(x, axis)
    obj.base(res=res, x=x, axis=axis)


@pytest.mark.api_base_unsqueeze__exception
def test_unsqueeze_1():
    """
    axis>max_dim
    """
    x = np.arange(6).reshape(2, 3)
    axis = 3
    obj.exception(mode="c", etype="InvalidArgument", x=x, axis=axis)


@pytest.mark.api_base_unsqueeze__parameters
def test_unsqueeze_2():
    """
    axis=negtive_num
    """
    x = np.arange(6).reshape(2, 3)
    axis = -1
    res = np.expand_dims(x, axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_unsqueeze__parameters
def test_unsqueeze_3():
    """
    axis=list
    """
    x = np.arange(6).reshape(2, 3)
    axis = [0, 1]
    res = np.expand_dims(x, axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_unsqueeze__parameters
def test_unsqueeze_4():
    """
    axis=tuple
    """
    x = np.arange(6).reshape(2, 3)
    axis = (0, 1)
    res = np.expand_dims(x, axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_unsqueeze__parameters
def test_unsqueeze_5():
    """
    axis=tensor
    """
    x = np.arange(6).reshape(2, 3)
    axis = np.array([0, 1, 2])
    res = np.expand_dims(x, axis=[0, 1, 2])
    obj.run(res=res, x=x, axis=axis)
