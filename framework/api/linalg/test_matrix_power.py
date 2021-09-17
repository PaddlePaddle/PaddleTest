#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test matrix_power
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestMatrixPower(APIBase):
    """
    test matrix_power
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestMatrixPower(paddle.linalg.matrix_power)


@pytest.mark.api_linalg_matrix_power_vartype
def test_matrix_power_base():
    """
    base
    """
    # x = randtool("float", -10, 10, (3, 3, 3))
    dim_n = 3
    x = np.random.random((3, 4, 4))
    res = np.linalg.matrix_power(x, dim_n)
    obj.base(res=res, x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_1():
    """
    np.float32
    x.shape=[3, 2, 2, 7, 6, 1, 11, 4, 4]
    n=3
    """
    dim_n = 3
    x = np.random.random((3, 2, 2, 7, 6, 1, 11, 4, 4)).astype(np.float32)
    res = np.linalg.matrix_power(x, dim_n)
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_2():
    """
    np.float32
    x.shape=[3, 2, 4, 4]
    n=8
    """
    dim_n = 8
    x = np.random.random((3, 2, 4, 4)).astype(np.float32)
    res = np.linalg.matrix_power(x, dim_n)
    obj.delta = 1e-5
    obj.rtol = 1e-6
    obj.run(res=res, x=x, n=dim_n)
    obj.delta = 1e-6
    obj.rtol = 1e-7


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_3():
    """
    np.float32
    x.shape=[3, 2, 4, 4]
    n=64
    """
    dim_n = 64
    x = np.random.random((3, 2, 4, 4)).astype(np.float32)
    res = np.linalg.matrix_power(x, dim_n)
    obj.delta = 1e-5
    obj.rtol = 1e-6
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True
    obj.delta = 1e-6
    obj.rtol = 1e-7


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_4():
    """
    np.float64
    x.shape=[3, 2, 10, 10]
    n=64
    """
    dim_n = 64
    x = np.random.random((3, 2, 10, 10)).astype(np.float64)
    res = np.linalg.matrix_power(x, dim_n)
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_5():
    """
    np.float64
    x.shape=[3, 2, 32, 32]
    n=10
    """
    dim_n = 10
    x = np.random.random((3, 2, 32, 32)).astype(np.float64)
    res = np.linalg.matrix_power(x, dim_n)
    obj.run(res=res, x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_6():
    """
    np.float64
    x.shape=[3, 2, 32, 32]
    n=0
    """
    dim_n = 0
    x = np.random.random((3, 2, 32, 32)).astype(np.float64)
    res = np.linalg.matrix_power(x, dim_n)
    obj.run(res=res, x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_7():
    """
    np.float64
    x.shape=[3, 2, 32, 32]
    n=-10
    """
    dim_n = -10
    x = np.random.random((3, 2, 32, 32)).astype(np.float64)
    res = np.linalg.matrix_power(x, dim_n)
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_8():
    """
    np.float64
    x.shape=[3, 2, 3, 2, 1, 32, 32]
    n=-10
    """
    dim_n = -10
    x = np.random.random((3, 2, 3, 2, 1, 32, 32)).astype(np.float64)
    res = np.linalg.matrix_power(x, dim_n)
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_9():
    """
    np.float32
    x.shape=[3, 2, 3, 2, 1, 32, 32]
    n=10
    """
    dim_n = 10
    x = np.random.random((3, 2, 3, 2, 1, 32, 32)).astype(np.float32)
    res = np.linalg.matrix_power(x, dim_n)
    obj.delta = 1e-5
    obj.rtol = 1e-6
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True
    obj.delta = 1e-6
    obj.rtol = 1e-7


@pytest.mark.api_linalg_matrix_power_parameters
def test_matrix_power_10():
    """
    np.float32
    x.shape=[3, 2, 3, 2, 1, 32, 32]
    n=-2
    """
    dim_n = -2
    x = np.random.random((3, 2, 3, 2, 1, 32, 32)).astype(np.float32)
    res = np.linalg.matrix_power(x, dim_n)
    obj.delta = 1e-3
    obj.rtol = 1e-4
    obj.enable_backward = False
    obj.run(res=res, x=x, n=dim_n)
    obj.enable_backward = True
    obj.delta = 1e-6
    obj.rtol = 1e-7


# @pytest.mark.api_linalg_matrix_power_parameters
# def test_matrix_power_11():
#     """
#     np.float32
#     x.shape=[3, 2, 3, 2, 1, 32, 32]
#     n=-10
#     """
#     dim_n = -10
#     x = np.random.random((3, 2, 3, 2, 1, 32, 32)).astype(np.float32)
#     res = np.linalg.matrix_power(x, dim_n)
#     obj.delta = 1e-1
#     obj.rtol = 1e-2
#     obj.enable_backward = False
#     obj.run(res=res, x=x, n=dim_n)
#     obj.enable_backward = True
#     obj.delta = 1e-6
#     obj.rtol = 1e-7


@pytest.mark.api_linalg_matrix_power_exception
def test_matrix_power_12():
    """
    TypeError
    x=42
    n=-2
    """
    dim_n = -2
    x = 42
    obj.exception(etype="TypeError", x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_exception
def test_matrix_power_13():
    """
    TypeError
    x.shape=[2, 1, 32, 32]
    n=[-2]
    """
    dim_n = [-2]
    x = np.random.random((2, 1, 32, 32)).astype(np.float32)
    obj.exception(etype="TypeError", x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_exception
def test_matrix_power_14():
    """
    TypeError
    x = np.array.astype(np.int32)
    n=-2
    """
    dim_n = -2
    x = np.random.random((2, 1, 32, 32)).astype(np.int32)
    obj.exception(etype="TypeError", x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_exception
def test_matrix_power_15():
    """
    ValueError
    x.shape=[32]
    n=-2
    """
    dim_n = -2
    x = np.random.random((32)).astype(np.float32)
    obj.exception(etype="ValueError", x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_exception
def test_matrix_power_16():
    """
    ValueError
    x.shape=[2, 1, 32, 16]
    n=-2
    """
    dim_n = -2
    x = np.random.random((2, 1, 32, 16)).astype(np.float32)
    obj.exception(etype="ValueError", x=x, n=dim_n)


@pytest.mark.api_linalg_matrix_power_exception
def test_matrix_power_17():
    """
    ValueError
    x=np.array([[2, 1, 2], [2, 1, 2], [1, 1, 7]]).astype(np.float32)
    n=-2
    """
    dim_n = -2
    x = np.array([[2, 1, 2], [2, 1, 2], [1, 1, 7]]).astype(np.float32)
    obj.exception(etype="ValueError", x=x, n=dim_n)
