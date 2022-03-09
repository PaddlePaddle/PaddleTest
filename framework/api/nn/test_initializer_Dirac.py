#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Dirac
"""

import numpy as np
import paddle
import pytest


def cal_dirac(shape):
    """
    calculate dirac init api
    """
    x = np.zeros(shape)
    N = min(shape[0], shape[1])

    for i in range(N):
        if len(shape) == 3:
            x[i, i, shape[2] // 2] = 1
        elif len(shape) == 4:
            x[i, i, shape[2] // 2, shape[3] // 2] = 1
        elif len(shape) == 5:
            x[i, i, shape[2] // 2, shape[3] // 2, shape[4] // 2] = 1

    return x


@pytest.mark.api_nn_Dirac_parameters
def test_dirac0():
    """
    conv1d
    """
    attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
    conv = paddle.nn.Conv1D(3, 4, 5, weight_attr=attr)
    res = cal_dirac((4, 3, 5))

    assert np.allclose(res, conv.weight.numpy())


@pytest.mark.api_nn_Dirac_parameters
def test_dirac1():
    """
    conv2d
    """
    attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
    conv = paddle.nn.Conv2D(3, 4, 5, weight_attr=attr)
    res = cal_dirac((4, 3, 5, 5))

    assert np.allclose(res, conv.weight.numpy())


@pytest.mark.api_nn_Dirac_parameters
def test_dirac2():
    """
    conv3d
    """
    attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
    conv = paddle.nn.Conv3D(3, 4, 5, weight_attr=attr)
    res = cal_dirac((4, 3, 5, 5, 5))

    assert np.allclose(res, conv.weight.numpy())
