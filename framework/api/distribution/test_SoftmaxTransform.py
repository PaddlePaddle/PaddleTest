#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_SoftmaxTransform
"""

from apibase import APIBase
import numpy as np
import paddle
import pytest


class TestSoftmaxTransform(APIBase):
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


def softmax_forward(x):
    """forward"""
    t = paddle.distribution.SoftmaxTransform()
    return t.forward(x)


def softmax_inverse(x):
    """
    inverse
    """
    t = paddle.distribution.SoftmaxTransform()
    return t.inverse(x)


obj0 = TestSoftmaxTransform(softmax_forward)
obj1 = TestSoftmaxTransform(softmax_inverse)


@pytest.mark.api_distribution_SoftmaxTransform_vartype
def test_SoftmaxTransform0():
    """
    base
    """
    x = np.ones((2, 3))
    res0 = x / np.sum(x, axis=1).reshape(2, 1)
    res1 = np.array([[-1.09861231, -1.09861231, -1.09861231], [-1.09861231, -1.09861231, -1.09861231]])
    obj0.base(res=res0, x=x)
    obj1.base(res=res1, x=res0)


@pytest.mark.api_distribution_SoftmaxTransform_parameters
def test_SoftmaxTransform1():
    """
    x：3-d tensor
    """
    x = np.ones((2, 3, 2))
    res0 = x / np.sum(x, axis=0)
    res1 = np.ones((2, 3, 2)) * -0.69314718
    obj0.run(res=res0, x=x)
    obj1.run(res=res1, x=res0)
