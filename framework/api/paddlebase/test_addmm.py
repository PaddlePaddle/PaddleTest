#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_addmm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAddmm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestAddmm(paddle.addmm)


@pytest.mark.api_base_addmm_vartype
def test_addmm_base():
    """
    base
    """
    x = np.ones((2, 2))
    y = np.ones((2, 2))
    data_input = np.ones((2, 2))
    alpha = 5.0
    beta = 0.5
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.base(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_parameters
def test_addmm():
    """
    default
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 5.0
    beta = 0.5
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_parameters
def test_addmm1():
    """
    alpha=0
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 0
    beta = 0.5
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_parameters
def test_addmm2():
    """
    alpha=-3.3
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = -3.3
    beta = 0.5
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_parameters
def test_addmm3():
    """
    alpha=-3.3, beta=0
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = -3.3
    beta = 0
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_parameters
def test_addmm4():
    """
    alpha = -3.3, beta=-0.7
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = -3.3
    beta = -0.7
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_parameters
def test_addmm5():
    """
    default alpha and beta
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 1
    beta = 1
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y)


@pytest.mark.api_base_addmm_parameters
def test_addmm6():
    """
    alpha=0, beta=0
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 0
    beta = 0
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm7():
    """
    exception x dim != 2
    """
    x = randtool("float", -10, 10, [5, 3, 4])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    obj.exception(etype=ValueError, mode="python", input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm8():
    """
    exception y dim != 2
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    obj.exception(etype=ValueError, mode="python", input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm9():
    """
    exception input dim != 2
    """
    x = randtool("float", -10, 10, [5, 3])
    y = randtool("float", -10, 10, [3, 5])
    data_input = randtool("float", -10, 10, [5, 5, 5, 5])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    obj.exception(etype=ValueError, mode="python", input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm10():
    """
    exception x[1] != y[0] 矩阵不可乘
    """
    x = randtool("float", -10, 10, [5, 2])
    y = randtool("float", -10, 10, [4, 5])
    data_input = randtool("float", -10, 10, [5, 5])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    obj.exception(etype=ValueError, mode="python", input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm11():
    """
    exception input shape != x dot y shape 矩阵不可加，不满足broadcast条件
    """
    x = randtool("float", -10, 10, [5, 4])
    y = randtool("float", -10, 10, [4, 5])
    data_input = randtool("float", -10, 10, [4, 4])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    obj.exception(etype=ValueError, mode="python", input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm12():
    """
    exception input shape != x dot y shape 矩阵可加，满足broadcast条件 input.shape = (1, 1)
    """
    x = randtool("float", -10, 10, [5, 4])
    y = randtool("float", -10, 10, [4, 5])
    data_input = randtool("float", -10, 10, [1, 1])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)


@pytest.mark.api_base_addmm_exception
def test_addmm13():
    """
    exception input shape != x dot y shape 矩阵可加，满足broadcast条件 input.shape = (5, 1)
    """
    x = randtool("float", -10, 10, [5, 4])
    y = randtool("float", -10, 10, [4, 5])
    data_input = randtool("float", -10, 10, [5, 1])
    alpha = 3.3
    beta = -3.3
    # 算法实现
    res = beta * data_input + alpha * np.dot(x, y)
    obj.run(res=res, input=data_input, x=x, y=y, beta=beta, alpha=alpha)
