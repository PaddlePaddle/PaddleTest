#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_StackTransform
"""

from apibase import APIBase
import numpy as np
import paddle
import pytest


class TestStackTransform(APIBase):
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
        self.delta = 1e-1


def stack_forward(x, **kwargs):
    """forward"""
    t = paddle.distribution.StackTransform(**kwargs)
    return t.forward(x)


def stack_inverse(x, **kwargs):
    """inverse"""
    t = paddle.paddle.distribution.StackTransform(**kwargs)
    return t.inverse(x)


def stack_forward_log_det_jacobian(x, **kwargs):
    """
    forward_log_det_jacobian
    """
    t = paddle.paddle.distribution.StackTransform(**kwargs)
    return t.forward_log_det_jacobian(x)


def stack_inverse_log_det_jacobian(x, **kwargs):
    """
    inverse_log_det_jacobian
    """
    t = paddle.paddle.distribution.StackTransform(**kwargs)
    return t.inverse_log_det_jacobian(x)


obj0 = TestStackTransform(stack_forward)
obj1 = TestStackTransform(stack_inverse)
obj2 = TestStackTransform(stack_forward_log_det_jacobian)
obj3 = TestStackTransform(stack_inverse_log_det_jacobian)


@pytest.mark.api_distribution_StackTransform_vartype
def test_StackTransform0():
    """
    base
    transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform())
    axis=1
    """
    x = np.stack((np.arange(1, 4), np.arange(1, 4)), 1)
    res0 = np.stack((np.exp(x[:, 0]), 1 / (1 + np.exp(-x[:, 1]))), 1)
    res1 = x
    res2 = np.array([[1.0, -1.62652326], [2.0, -2.25385618], [3.0, -3.09717464]])
    res3 = -res2
    obj0.base(
        res=res0, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()), axis=1
    )
    obj1.base(
        res=res1,
        x=res0,
        transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()),
        axis=1,
    )
    obj2.base(
        res=res2, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()), axis=1
    )
    obj3.base(
        res=res3,
        x=res0,
        transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()),
        axis=1,
    )


@pytest.mark.api_distribution_StackTransform_parameters
def test_StackTransform1():
    """
    transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform())
    axis=0
    """
    x = np.stack((np.arange(1, 4), np.arange(1, 4)), 0)
    res0 = np.stack((np.exp(np.arange(1, 4)), 1 / (1 + np.exp(-np.arange(1, 4)))), 0)
    res1 = x
    res2 = np.array([[1.0, 2.0, 3.0], [-1.62652326, -2.25385618, -3.09717464]])
    res3 = -res2
    obj0.run(
        res=res0, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()), axis=0
    )
    obj1.run(
        res=res1,
        x=res0,
        transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()),
        axis=0,
    )
    obj2.run(
        res=res2, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()), axis=0
    )
    obj3.run(
        res=res3,
        x=res0,
        transforms=(paddle.distribution.ExpTransform(), paddle.distribution.SigmoidTransform()),
        axis=0,
    )


@pytest.mark.api_distribution_StackTransform_parameters
def test_StackTransform2():
    """
    transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform())
    axis=0
    """
    obj0.enable_backward = obj1.enable_backward = obj2.enable_backward = obj3.enable_backward = False
    x = np.stack((np.arange(1, 4), np.arange(1, 4)), 0)
    res0 = np.stack((np.exp(np.arange(1, 4)), np.tanh(np.arange(1, 4))), 0)
    res1 = x
    res2 = np.array([[1.0, 2.0, 3.0], [-0.86756170, -2.65000558, -4.61865711]])
    res3 = -res2
    obj0.run(
        res=res0, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=0
    )
    obj1.run(
        res=res1, x=res0, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=0
    )
    obj2.run(
        res=res2, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=0
    )
    obj3.run(
        res=res3, x=res0, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=0
    )


@pytest.mark.api_distribution_StackTransform_parameters
def test_StackTransform3():
    """
    transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform())
    axis=1
    """
    obj0.enable_backward = obj1.enable_backward = obj2.enable_backward = obj3.enable_backward = False
    x = np.stack((np.arange(1, 4), np.arange(1, 4)), 1)
    res0 = np.stack((np.exp(np.arange(1, 4)), np.tanh(np.arange(1, 4))), 1)
    res1 = x
    res2 = np.array([[1.0, -0.86756170], [2.0, -2.65000558], [3.0, -4.61865711]])
    res3 = -res2
    obj0.run(
        res=res0, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=1
    )
    obj1.run(
        res=res1, x=res0, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=1
    )
    obj2.run(
        res=res2, x=x, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=1
    )
    obj3.run(
        res=res3, x=res0, transforms=(paddle.distribution.ExpTransform(), paddle.distribution.TanhTransform()), axis=1
    )
