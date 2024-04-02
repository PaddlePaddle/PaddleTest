#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_TanhTransform
"""

from apibase import APIBase
import numpy as np
import paddle
import pytest


class TestTanhTransform(APIBase):
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
        self.delta = 1e-3


def tanh_forward(x):
    """forward"""
    tanh = paddle.distribution.TanhTransform()
    return tanh.forward(x)


def tanh_inverse(x):
    """inverse"""
    tanh = paddle.distribution.TanhTransform()
    return tanh.inverse(x)


def tanh_forward_log_det_jacobian(x):
    """
    forward_log_det_jacobian
    """
    tanh = paddle.distribution.TanhTransform()
    return tanh.forward_log_det_jacobian(x)


def tanh_inverse_log_det_jacobian(x):
    """
    inverse_log_det_jacobian
    """
    tanh = paddle.distribution.TanhTransform()
    return tanh.inverse_log_det_jacobian(x)


obj0 = TestTanhTransform(tanh_forward)
obj1 = TestTanhTransform(tanh_inverse)
obj2 = TestTanhTransform(tanh_forward_log_det_jacobian)
obj3 = TestTanhTransform(tanh_inverse_log_det_jacobian)


@pytest.mark.api_distribution_TanhTransform_vartype
def test_TanhTransform0():
    """
    base
    """
    x = np.array([1.0, 2.0, 3.0])
    res0 = np.tanh(x)
    res1 = x
    res2 = np.array([-0.86756170, -2.65000558, -4.61865711])
    res3 = -res2
    obj0.base(res=res0, x=x)
    obj1.base(res=res1, x=res0)
    obj2.base(res=res2, x=x)
    obj3.base(res=res3, x=res0)


@pytest.mark.api_distribution_TanhTransform_vartype
def test_TanhTransform1():
    """
    x: 2-d tensor
    """
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    res0 = np.tanh(x)
    res1 = x
    res2 = np.array([[-0.86756170, -2.65000558, -4.61865711], [-6.61437654, -8.61379623, -10.61371803]])
    res3 = -res2
    obj0.base(res=res0, x=x)
    obj1.base(res=res1, x=res0)
    obj2.base(res=res2, x=x)
    obj3.base(res=res3, x=res0)


@pytest.mark.api_distribution_TanhTransform_vartype
def test_TanhTransform2():
    """
    x: 3-d tensor
    """
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    res0 = np.tanh(x)
    res1 = x
    res2 = np.array([[[-0.86756170, -2.65000558, -4.61865711], [-6.61437654, -8.61379623, -10.61371803]]])
    res3 = -res2
    obj0.base(res=res0, x=x)
    obj1.base(res=res1, x=res0)
    obj2.base(res=res2, x=x)
    obj3.base(res=res3, x=res0)
