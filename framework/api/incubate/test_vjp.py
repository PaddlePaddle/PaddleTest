#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_vjp
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np
from fdu_base import FiniteDifferenceUtils


class TestVjp(APIBase):
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


def cal_vjp(func, xs, v=None):
    """
    vjp API
    """
    vjp = paddle.incubate.autograd.vjp(func, xs, v)[1]
    if isinstance(vjp, paddle.fluid.framework.Variable):
        return vjp.reshape((-1,))
    else:
        return paddle.concat([x.reshape((-1,)) for x in vjp]).reshape((-1,))


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


obj = TestVjp(cal_vjp)
ans = FiniteDifferenceUtils()


@pytest.mark.api_incubate_vjp_vartype
def test_vjp_base():
    """
    base
    x: 1-d tensor
    single_input and single_output
    """
    x = np.random.rand(2)
    paddle.disable_static()
    res = ans.vjp_with_jac(func0, paddle.to_tensor(x))
    obj.base(res=res, func=func0, xs=x)


@pytest.mark.api_incubate_vjp_parameters
def test_vjp1():
    """
    x: 2-d tensor
    single_input and single_output
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    paddle.disable_static()
    res = ans.vjp_with_jac(func0, paddle.to_tensor(x))
    obj.run(res=res, func=func0, xs=x)


@pytest.mark.api_incubate_vjp_parameters
def test_vjp2():
    """
    x: 2-d tensor
    single_input and single_output
    set v
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.random.rand(2, 2)
    paddle.disable_static()
    res = ans.vjp_with_jac(func0, paddle.to_tensor(x), v=paddle.to_tensor(v))
    obj.run(res=res, func=func0, xs=x, v=v)


@pytest.mark.api_incubate_vjp_parameters
def test_vjp3():
    """
    x: 2-d tensor
    multi_input and single_output
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    paddle.disable_static()
    res = ans.vjp_with_jac(func1, [paddle.to_tensor(x), paddle.to_tensor(x)])
    obj.run(res=res, func=func1, xs=[x, x])


@pytest.mark.api_incubate_vjp_parameters
def test_vjp4():
    """
    x: 2-d tensor
    multi_input and single_output
    set v
    """
    x = np.random.rand(3, 3)
    v = np.random.rand(3, 3)
    paddle.disable_static()
    res = ans.vjp_with_jac(func1, [paddle.to_tensor(x), paddle.to_tensor(x)], v=paddle.to_tensor(v))
    obj.run(res=res, func=func1, xs=[x, x], v=v)


@pytest.mark.api_incubate_vjp_parameters
def test_vjp5():
    """
    x: 2-d tensor
    single_input and multi_output
    """
    x = np.random.rand(3, 3)
    paddle.disable_static()
    res = ans.vjp_with_jac(func2, paddle.to_tensor(x))
    obj.run(res=res, func=func2, xs=x)


@pytest.mark.api_incubate_vjp_parameters
def test_vjp6():
    """
    x: 2-d tensor
    single_input and multi_output
    set v
    """
    x = np.random.rand(3, 3)
    v = np.random.rand(3, 3)
    paddle.disable_static()
    res = ans.vjp_with_jac(func2, paddle.to_tensor(x), v=[paddle.to_tensor(v), paddle.to_tensor(v)])
    obj.run(res=res, func=func2, xs=x, v=[v, v])


@pytest.mark.api_incubate_vjp_parameters
def test_vjp7():
    """
    x: 2-d tensor
    multi_input and multi_output
    """
    x = np.random.rand(3, 3)
    y = np.random.rand(3, 3)
    paddle.disable_static()
    res = ans.vjp_with_jac(func3, [paddle.to_tensor(x), paddle.to_tensor(y)])
    obj.run(res=res, func=func3, xs=[x, y])


@pytest.mark.api_incubate_vjp_parameters
def test_vjp8():
    """
    x: 2-d tensor
    multi_input and multi_output
    set v
    """
    x = np.random.rand(4, 4)
    v1 = np.random.rand(4, 4)
    paddle.disable_static()
    res = ans.vjp_with_jac(
        func3, [paddle.to_tensor(x), paddle.to_tensor(x)], v=[paddle.to_tensor(v1), paddle.to_tensor(v1)]
    )
    obj.run(res=res, func=func3, xs=[x, x], v=[v1, v1])
