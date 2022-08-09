#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_lu
"""
import numpy as np
import paddle
import scipy.linalg
import pytest
from apibase import randtool


def cal_lu(x):
    """
    calculate lu
    """
    shape = x.shape
    if len(shape) == 2:
        p, l, u = scipy.linalg.lu(x)
        d = np.eye(len(x[0]))
        lu = l + u - d
        return lu
    else:
        batchsize = np.product(shape) // (shape[-2] * shape[-1])
        nx = x.reshape((-1, shape[-2], shape[-1]))
        tlu = []
        for i in range(batchsize):
            p, l, u = scipy.linalg.lu(nx[i])
            d = np.eye(shape[-1])
            lu = l + u - d
            tlu.append(lu)
        return np.array(tlu).reshape(shape)


def cal_api(x, dtype="float64"):
    """
    calculate lu api
    """
    x = paddle.to_tensor(x, dtype=dtype)
    return paddle.linalg.lu(x)[0]


@pytest.mark.api_linalg_lu_vartype
def test_lu_base():
    """
    base
    """
    x = randtool("float", -2, 2, (3, 3))
    for dtype in ["float32", "float64"]:
        res = cal_lu(x)
        p_res = cal_api(x, dtype)
        assert np.allclose(res, p_res.numpy())


@pytest.mark.api_linalg_lu_parameters
def test_lu0():
    """
    x: 3d tensor
    """
    x = randtool("float", -2, 2, (3, 3, 3))
    res = cal_lu(x)
    p_res = cal_api(x)
    assert np.allclose(res, p_res.numpy())


@pytest.mark.api_linalg_lu_parameters
def test_lu1():
    """
    x: 4d tensor
    """
    x = randtool("float", -2, 2, (4, 3, 2, 2))
    res = cal_lu(x)
    p_res = cal_api(x)
    assert np.allclose(res, p_res.numpy())
