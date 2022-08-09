#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ReshapeTransform
"""

from apibase import APIBase
import numpy as np
import paddle
import pytest


class TestReshapeTransform(APIBase):
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
        # self.delta = 1e-3


def reshape_forward(x, in_event_shape, out_event_shape):
    """forward"""
    reshape_transform = paddle.distribution.ReshapeTransform(in_event_shape, out_event_shape)
    return reshape_transform.forward(x)


def reshape_inverse(x, in_event_shape, out_event_shape):
    """inverse"""
    reshape_transform = paddle.distribution.ReshapeTransform(in_event_shape, out_event_shape)
    return reshape_transform.inverse(x)


def reshape_forward_log_det_jacobian(x, in_event_shape, out_event_shape):
    """
    forward_log_det_jacobian
    """
    reshape_transform = paddle.distribution.ReshapeTransform(in_event_shape, out_event_shape)
    return reshape_transform.forward_log_det_jacobian(x)


def reshape_inverse_log_det_jacobian(x, in_event_shape, out_event_shape):
    """
    inverse_log_det_jacobian
    """
    reshape_transform = paddle.distribution.ReshapeTransform(in_event_shape, out_event_shape)
    return reshape_transform.inverse_log_det_jacobian(x)


obj0 = TestReshapeTransform(reshape_forward)
obj1 = TestReshapeTransform(reshape_inverse)
obj2 = TestReshapeTransform(reshape_forward_log_det_jacobian)
obj3 = TestReshapeTransform(reshape_inverse_log_det_jacobian)
obj2.enable_backward = False
obj3.enable_backward = False


@pytest.mark.api_distribution_ReshapeTransform_vartype
def test_ReshapeTransform0():
    """
    in_event_shape=(2, 3)
    out_event_shape=(3, 2)
    """
    x = np.ones((1, 2, 3))
    res0 = np.ones((1, 3, 2))
    res1 = np.ones((1, 2, 3))
    res2 = res3 = np.array([0.0])
    obj0.base(res=res0, x=x, in_event_shape=(2, 3), out_event_shape=(3, 2))
    obj1.base(res=res1, x=res0, in_event_shape=(2, 3), out_event_shape=(3, 2))
    obj2.base(res=res2, x=x, in_event_shape=(2, 3), out_event_shape=(3, 2))
    obj3.base(res=res3, x=x, in_event_shape=(2, 3), out_event_shape=(3, 2))


@pytest.mark.api_distribution_ReshapeTransform_parameters
def test_ReshapeTransform1():
    """
    in_event_shape=(2, 3)
    out_event_shape=(3, 2)
    """
    x = np.ones((1, 2, 3))
    res0 = np.ones((1, 3, 2))
    res1 = np.ones((1, 2, 3))
    res2 = res3 = np.array([0.0])
    obj0.run(res=res0, x=x, in_event_shape=(2, 3), out_event_shape=(3, 2))
    obj1.run(res=res1, x=res0, in_event_shape=(2, 3), out_event_shape=(3, 2))
    obj2.run(res=res2, x=x, in_event_shape=(2, 3), out_event_shape=(3, 2))
    obj3.run(res=res3, x=x, in_event_shape=(2, 3), out_event_shape=(3, 2))


@pytest.mark.api_distribution_ReshapeTransform_parameters
def test_ReshapeTransform2():
    """
    in_event_shape=(1, 2, 3)
    out_event_shape=(3, 1, 2)
    """
    x = np.ones((1, 2, 3))
    res0 = np.ones((3, 1, 2))
    res1 = np.ones((1, 2, 3))
    res2 = res3 = np.array([0.0])
    obj0.run(res=res0, x=x, in_event_shape=(1, 2, 3), out_event_shape=(3, 1, 2))
    obj1.run(res=res1, x=res0, in_event_shape=(1, 2, 3), out_event_shape=(3, 1, 2))
    obj2.run(res=res2, x=x, in_event_shape=(1, 2, 3), out_event_shape=(3, 1, 2))
    obj3.run(res=res3, x=x, in_event_shape=(1, 2, 3), out_event_shape=(3, 1, 2))
