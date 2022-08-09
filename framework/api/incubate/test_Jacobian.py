#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_jacbian
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np
from fdu_base import FiniteDifferenceUtils


class TestJacbian(APIBase):
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


def cal_jacbian(func, inputs, is_batched=False):
    """
    Jacobin API
    """
    J = paddle.incubate.autograd.Jacobian(func, inputs, is_batched=is_batched)
    return J[:]


def func0(x):
    """
    func0
    """
    return x * x


def func1(x, y):
    """
    func1
    """
    return paddle.matmul(x, y)


def func2(x):
    """
    func2
    """
    return paddle.matmul(x, x), x * x


def func3(x, y):
    """
    func3
    """
    return paddle.tanh(x), x * y


def func4(x, y, z):
    """
    func4
    """
    return paddle.tanh(x), x * y, paddle.matmul(x, z)


def func5(x, y, z):
    """
    func5
    """
    return paddle.tanh(x), x * y, paddle.tanh(z)


obj = TestJacbian(cal_jacbian)
ans = FiniteDifferenceUtils()


@pytest.mark.api_incubate_jacbian_vartype
def test_jacbian_base():
    """
    base
    x: 1-d tensor
    single_input and single_output
    """
    x = np.random.rand(2)
    paddle.disable_static()
    res = ans.numerical_jacobian(func0, paddle.to_tensor(x))
    paddle.enable_static()
    obj.base(res=res, func=func0, inputs=x)


@pytest.mark.api_incubate_jacbian_vartype
def test_jacbian1():
    """
    x: 2-d tensor
    multi_input and single_output
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    paddle.disable_static()
    res = ans.numerical_jacobian(func1, [paddle.to_tensor(x), paddle.to_tensor(x)])
    obj.run(res=res, func=func1, inputs=[x, x])


@pytest.mark.api_incubate_jacbian_parameters
def test_jacbian2():
    """
    x: 2-d tensor
    single_input and multi_output
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    paddle.disable_static()
    res = ans.numerical_jacobian(func2, paddle.to_tensor(x))
    obj.run(res=res, func=func2, inputs=x)


@pytest.mark.api_incubate_jacbian_parameters
def test_jacbian3():
    """
    x: 2-d tensor
    multi_input and multi_output
    """
    x = np.random.rand(2, 3)
    paddle.disable_static()
    res = ans.numerical_jacobian(func3, [paddle.to_tensor(x), paddle.to_tensor(x)])
    obj.run(res=res, func=func3, inputs=[x, x])


@pytest.mark.api_incubate_jacbian_parameters
def test_jacbian4():
    """
    x: 2-d tensor
    multi_input and multi_output
    """
    x = np.random.rand(2, 3)
    paddle.disable_static()
    res = ans.numerical_jacobian(func4, [paddle.to_tensor(x), paddle.to_tensor(x), paddle.to_tensor(x.T)])
    obj.run(res=res, func=func4, inputs=[x, x, x.T])


@pytest.mark.api_incubate_jacbian_parameters
def test_jacbian5():
    """
    x: 3-d tensor
    multi_input and multi_output
    """
    x = np.random.rand(2, 3, 4)
    paddle.disable_static()
    res = ans.numerical_jacobian(func5, [paddle.to_tensor(x), paddle.to_tensor(x), paddle.to_tensor(x)])
    obj.run(res=res, func=func5, inputs=[x, x, x])


@pytest.mark.api_incubate_jacbian_parameters
def test_jacbian6():
    """
    x: 2-d tensor
    multi_input and single_output
    is_batch=true
    """
    x = np.random.rand(2, 3, 4)
    paddle.disable_static()
    res = ans.numerical_jacobian(func0, paddle.to_tensor(x), is_batch=True)
    obj.run(res=res, func=func0, inputs=x, is_batched=True)


@pytest.mark.api_incubate_jacbian_parameters
def test_jacbian7():
    """
    x: 2-d tensor
    multi_input and single_output
    is_batch=true
    """
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)
    paddle.disable_static()
    res = ans.numerical_jacobian(func3, [paddle.to_tensor(x), paddle.to_tensor(y)], is_batch=True)
    obj.run(res=res, func=func3, inputs=[x, y], is_batched=True)
