#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_add_n
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestAddn(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-3


obj = TestAddn(paddle.add_n)


@pytest.mark.api_base_add_n_vartype
def test_add_n_base():
    """
    base
    """
    x = np.random.randint(-4, 4, (10,))
    res = x
    obj.base(res=res, inputs=x)


@pytest.mark.api_base_add_n_parameters
def test_add_n0():
    """
    default
    """
    x = np.random.rand(4, 4)
    res = x
    obj.run(res=res, inputs=x)


@pytest.mark.api_base_add_n_parameters
def test_add_n1():
    """
    x: 3d - tensor
    """
    x = np.random.rand(4, 4, 4)
    res = x
    obj.run(res=res, inputs=x)


@pytest.mark.api_base_add_n_parameters
def test_add_n2():
    """
    x: 4d - tensor
    """
    x = np.random.rand(4, 4, 4, 4)
    res = x
    obj.run(res=res, inputs=x)


@pytest.mark.api_base_add_n_parameters
def test_add_n3():
    """
    input: list
    """
    paddle.disable_static()
    x = np.random.rand(4, 4) * 4
    y = np.random.rand(4, 4) * 2
    res = x + y
    api_res = paddle.add_n([paddle.to_tensor(x), paddle.to_tensor(y)])
    assert np.allclose(res, api_res.numpy())


@pytest.mark.api_base_add_n_parameters
def test_add_n4():
    """
    input: list
    """
    paddle.disable_static()
    x = np.random.rand(4, 4) * 4
    y = np.random.rand(4, 4) * 2
    res = x + x + y
    api_res = paddle.add_n([paddle.to_tensor(x), paddle.to_tensor(x), paddle.to_tensor(y)])
    assert np.allclose(res, api_res.numpy())
