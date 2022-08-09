#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_StickBreakingTransform
"""

from apibase import APIBase
import numpy as np
import paddle
import pytest


class TestStickBreakingTransform(APIBase):
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
        # self.delta = 1e-1


def stickbreaking_forward(x):
    """forward"""
    t = paddle.distribution.StickBreakingTransform()
    return t.forward(x)


def stickbreaking_inverse(x):
    """inverse"""
    t = paddle.distribution.StickBreakingTransform()
    return t.inverse(x)


def stickbreaking_forward_log_det_jacobian(x):
    """
    forward_log_det_jacobian
    """
    t = paddle.distribution.StickBreakingTransform()
    return t.forward_log_det_jacobian(x)


def stickbreaking_inverse_log_det_jacobian(x):
    """
    inverse_log_det_jacobian
    """
    t = paddle.distribution.StickBreakingTransform()
    return t.inverse_log_det_jacobian(x)


obj0 = TestStickBreakingTransform(stickbreaking_forward)
obj1 = TestStickBreakingTransform(stickbreaking_inverse)
obj2 = TestStickBreakingTransform(stickbreaking_forward_log_det_jacobian)
obj3 = TestStickBreakingTransform(stickbreaking_inverse_log_det_jacobian)


def cal_forward(x):
    """
    calculate StickBreakingTransform forward
    """
    offset = x.shape[-1] + 1 - np.ones([x.shape[-1]]).cumsum(-1)
    z = 1 / (1 + np.exp(np.log(offset) - x))
    z_cumprod = (1 - z).cumprod(-1)
    print(z_cumprod)
    return np.pad(z, (0, 1), constant_values=1) * np.pad(z_cumprod, (1, 0), constant_values=1)


@pytest.mark.api_distribution_StickBreakingTransform_vartype
def test_StickBreakingTransform0():
    """
    base
    """
    x = np.array([1.0, 2.0, 3.0])
    res0 = cal_forward(x)
    res1 = x
    res2 = np.array([-9.10835171])
    res3 = np.array([9.10835075])
    obj0.base(res=res0, x=x)
    obj1.base(res=res1, x=res0)
    obj2.base(res=res2, x=x)
    obj3.base(res=res3, x=res0)
