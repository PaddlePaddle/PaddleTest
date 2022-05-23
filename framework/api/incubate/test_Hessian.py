#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Hessian
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np
from fdu_base import FiniteDifferenceUtils, _np_transpose_matrix_format, MatrixFormat


class TestHessian(APIBase):
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
        self.delta = 1e-5


def cal_hessian(func, inputs, is_batched=False):
    """
    hessian API
    """

    h = paddle.incubate.autograd.Hessian(func, inputs, is_batched=is_batched)
    if not is_batched:
        return h[:]
    else:
        actual = _np_transpose_matrix_format(h[:].numpy(), MatrixFormat.BNM, MatrixFormat.NBM)
        return paddle.to_tensor(actual.reshape((h.shape[1], -1)))


def func0(x):
    """
    func0
    """
    return paddle.sum(x * x)


def func1(x, y):
    """
    func1
    """
    return paddle.sum(paddle.matmul(x, y))


def func2(x):
    """
    func1
    """
    return paddle.sum(paddle.nn.functional.sigmoid(x))


obj = TestHessian(cal_hessian)
ans = FiniteDifferenceUtils()


@pytest.mark.api_incubate_Hessian_vartype
def test_Hessian_base():
    """
    base
    x: 1-d tensor
    single_input
    """
    x = np.random.rand(2)
    paddle.disable_static()
    res = ans.numerical_hessian(func0, paddle.to_tensor(x, dtype="float64"), is_batched=False)
    paddle.enable_static()
    obj.enable_backward = False
    obj.base(res=res, func=func0, inputs=x)


@pytest.mark.api_incubate_Hessian_parameters
def test_Hessian1():
    """
    base
    x: 2-d tensor
    single_input
    """
    x = np.random.rand(2, 3)
    paddle.disable_static()
    res = ans.numerical_hessian(func0, paddle.to_tensor(x, dtype="float64"), is_batched=False)
    paddle.enable_static()
    obj.enable_backward = False
    obj.run(res=res, func=func0, inputs=x)


@pytest.mark.api_incubate_Hessian_parameters
def test_Hessian2():
    """
    base
    x: 4-d tensor
    single_input
    """
    x = np.random.rand(2, 3, 3, 2)
    paddle.disable_static()
    res = ans.numerical_hessian(func0, paddle.to_tensor(x, dtype="float64"), is_batched=False)
    paddle.enable_static()
    obj.enable_backward = False
    obj.run(res=res, func=func0, inputs=x)


@pytest.mark.api_incubate_Hessian_parameters
def test_Hessian3():
    """
    base
    x: 2-d tensor
    multiple_input
    """
    x = np.random.rand(2, 3)
    y = np.random.rand(3, 2)
    paddle.disable_static()
    res = ans.numerical_hessian(
        func1, [paddle.to_tensor(x, dtype="float64"), paddle.to_tensor(y, dtype="float64")], is_batched=False
    )
    paddle.enable_static()
    obj.enable_backward = False
    obj.run(res=res, func=func1, inputs=[x, y])


@pytest.mark.api_incubate_Hessian_parameters
def test_Hessian4():
    """
    base
    x: 2-d tensor
    single_input
    """
    x = np.random.rand(4, 3)
    paddle.disable_static()
    res = ans.numerical_hessian(func2, paddle.to_tensor(x, dtype="float64"), is_batched=True)
    paddle.enable_static()
    obj.static = False
    obj.enable_backward = False
    obj.run(res=res, func=func2, inputs=x, is_batched=True)
