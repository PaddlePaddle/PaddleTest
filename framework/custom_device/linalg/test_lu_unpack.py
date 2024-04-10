#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_lu_unpack
"""
import numpy as np
import paddle
import scipy.linalg
import pytest
from apibase import randtool


def cal_lu(x):
    """
    calculate lu_unpack
    """
    shape = x.shape
    if len(shape) == 2:
        p, l, u = scipy.linalg.lu(x)
        return p, l, u
    else:
        batchsize = np.product(shape) // (shape[-2] * shape[-1])
        nx = x.reshape((-1, shape[-2], shape[-1]))
        tp, tl, tu = [], [], []
        for i in range(batchsize):
            p, l, u = scipy.linalg.lu(nx[i])
            tp.append(p)
            tl.append(l)
            tu.append(u)
        return np.array(tp).reshape(shape), np.array(tl).reshape(shape), np.array(tu).reshape(shape)


def cal_api(x, dtype="float64"):
    """
    calculate lu_unpack api
    """
    x = paddle.to_tensor(x, dtype=dtype)
    lu, p = paddle.linalg.lu(x, get_infos=False)
    p, l, u = paddle.linalg.lu_unpack(lu, p)
    return p, l, u


@pytest.mark.api_linalg_lu_unpack_vartype
def test_lu_unpack_base():
    """
    base
    """
    x = randtool("float", -2, 2, (3, 3))
    for dtype in ["float32", "float64"]:
        res = cal_lu(x)
        p_res = cal_api(x, dtype)
        for i in range(3):
            assert np.allclose(res[i], p_res[i].numpy())


@pytest.mark.api_linalg_lu_unpack_parameters
def test_lu_unpack0():
    """
    x: 3d tensor
    """
    x = randtool("float", -2, 2, (3, 3, 3))
    res = cal_lu(x)
    p_res = cal_api(x)
    for i in range(3):
        assert np.allclose(res[i], p_res[i].numpy())


@pytest.mark.api_linalg_lu_unpack_parameters
def test_lu_unpack1():
    """
    x: 3d tensor
    """
    x = randtool("float", -2, 2, (5, 3, 3, 3))
    res = cal_lu(x)
    p_res = cal_api(x)
    for i in range(3):
        assert np.allclose(res[i], p_res[i].numpy())
