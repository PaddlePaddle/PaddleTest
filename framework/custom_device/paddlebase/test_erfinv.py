#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_erfinv
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


def create_ins(xp):
    """
    create inputs
    """
    paddle.disable_static()
    x = paddle.to_tensor(xp)
    ins = paddle.erf(x).numpy()
    return ins


class TestErfinv(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-3


obj = TestErfinv(paddle.erfinv)


@pytest.mark.api_base_erfinv_vartype
def test_erfinv_base():
    """
    base
    """
    data = randtool("float", 0, 1, (4,))
    x = create_ins(data)
    obj.base(res=data, x=x)


@pytest.mark.api_base_erfinv_parameters
def test_erfinv0():
    """
    x: 2d-tensor
    """
    data = randtool("float", -2, 2, (4, 2))
    x = create_ins(data)
    obj.run(res=data, x=x)


@pytest.mark.api_base_erfinv_parameters
def test_erfinv1():
    """
    x: 3d-tensor
    """
    data = randtool("float", -2, 2, (4, 2, 3))
    x = create_ins(data)
    obj.run(res=data, x=x)


@pytest.mark.api_base_erfinv_parameters
def test_erfinv2():
    """
    x: 4d-tensor
    """
    data = randtool("float", -2, 2, (4, 2, 3, 5))
    x = create_ins(data)
    obj.run(res=data, x=x)


@pytest.mark.api_base_erfinv_parameters
def test_erfinv3():
    """
    x: 5d-tensor
    """
    data = randtool("float", -2, 2, (4, 2, 3, 5, 4))
    x = create_ins(data)
    obj.run(res=data, x=x)
