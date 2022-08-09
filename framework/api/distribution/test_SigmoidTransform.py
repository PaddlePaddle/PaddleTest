#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_SigmoidTransform
"""

from apibase import APIBase
import numpy as np
import paddle
import pytest


class TestSigmoidTransform(APIBase):
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
        # self.enable_backward = False
        # self.delta = 1e-1


def sigmoid_forward(x):
    """forward"""
    t = paddle.distribution.SigmoidTransform()
    return t.forward(x)


def sigmoid_inverse(x):
    """inverse"""
    t = paddle.distribution.SigmoidTransform()
    return t.inverse(x)


def sigmoid_forward_log_det_jacobian(x):
    """
    forward_log_det_jacobian
    """
    t = paddle.distribution.SigmoidTransform()
    return t.forward_log_det_jacobian(x)


def sigmoid_inverse_log_det_jacobian(x):
    """
    inverse_log_det_jacobian
    """
    t = paddle.distribution.SigmoidTransform()
    return t.inverse_log_det_jacobian(x)


obj0 = TestSigmoidTransform(sigmoid_forward)
obj1 = TestSigmoidTransform(sigmoid_inverse)
obj2 = TestSigmoidTransform(sigmoid_forward_log_det_jacobian)
obj3 = TestSigmoidTransform(sigmoid_inverse_log_det_jacobian)


@pytest.mark.api_distribution_SigmoidTransform_vartype
def test_SigmoidTransform0():
    """
    base
    """
    x = np.ones((2, 3))
    res0 = 1 / (1 + np.exp(-x))
    res1 = np.ones((2, 3))
    res2 = np.array([[-1.62652326, -1.62652326, -1.62652326], [-1.62652326, -1.62652326, -1.62652326]])
    res3 = -np.array([[-1.62652326, -1.62652326, -1.62652326], [-1.62652326, -1.62652326, -1.62652326]])
    obj0.base(res=res0, x=x)
    obj1.base(res=res1, x=res0)
    obj2.base(res=res2, x=x)
    obj3.base(res=res3, x=res0)


@pytest.mark.api_distribution_SigmoidTransform_parameters
def test_SigmoidTransform1():
    """
    x：3-d tensor
    """
    x = np.ones((2, 3, 2))
    res0 = 1 / (1 + np.exp(-x))
    res1 = np.ones((2, 3, 2))
    res2 = np.array(
        [
            [[-1.62652326, -1.62652326], [-1.62652326, -1.62652326], [-1.62652326, -1.62652326]],
            [[-1.62652326, -1.62652326], [-1.62652326, -1.62652326], [-1.62652326, -1.62652326]],
        ]
    )
    res3 = -res2
    obj0.run(res=res0, x=x)
    obj1.run(res=res1, x=res0)
    obj2.run(res=res2, x=x)
    obj3.run(res=res3, x=res0)


@pytest.mark.api_distribution_SigmoidTransform_parameters
def test_SigmoidTransform2():
    """
    x：1-d tensor
    """
    x = np.array([1.0, 2.0, 3.0])
    res0 = 1 / (1 + np.exp(-x))
    res1 = np.array([1.0, 2.0, 3.0])
    res2 = np.array([-1.62652326, -2.25385618, -3.09717464])
    res3 = np.array([1.62652338, 2.25385571, 3.09717488])
    obj0.delta = obj2.delta = obj1.delta = obj3.delta = 1e-1
    obj0.run(res=res0, x=x)
    obj1.run(res=res1, x=res0)
    obj2.run(res=res2, x=x)
    obj3.run(res=res3, x=res0)
