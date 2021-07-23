#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test diag
"""
from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestDiag(APIBase):
    """
    test diag
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # enable check grad
        self.no_grad_var = "offset"
        self.enable_backward = False


obj = TestDiag(paddle.diag)


@pytest.mark.api_nn_diag_vartype
def test_diag_base():
    """
    base,x=1-D
    """
    x = np.array([2, 3, 4])
    res = np.diag(x)
    obj.base(res=res, x=x)


# test x
@pytest.mark.api_base_diag_parameters
def test_diag():
    """
    x=2-D,x_shape=[3,3]
    """
    x = randtool("float", 1, 10, [3, 3])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag1():
    """
    x=1-D,x_shape=[1]
    """
    x = randtool("float", -10, 10, [1])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag2():
    """
    x=1-D,x_shape=[2]
    """
    x = randtool("float", -10, 10, [2])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag3():
    """
    x=1-D,x_shape=[3]
    """
    x = randtool("float", -10, 10, [3])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag4():
    """
    x=1-D,x = np.zeros([1])
    """
    x = np.zeros([1])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag5():
    """
    x=2-D,x_shape=[3,2]
    """
    x = randtool("float", 1, 10, [3, 2])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag6():
    """
    x=2-D,x_shape=[3,1]
    """
    x = randtool("float", -10, 10, [3, 1])
    res = np.diag(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diag_parameters
def test_diag7():
    """
    x=2-D,x_shape=[3,2]
    """
    x = np.zeros([3, 2])
    res = np.diag(x)
    obj.run(res=res, x=x)


# test offset, if X=2-D,offset=[-rank(X),rank(X))
@pytest.mark.api_base_diag_parameters
def test_diag8():
    """
    x=2-D,offset=0,x_shape=[3,3]
    """
    x = randtool("float", -10, 10, [3, 3])
    offset = 0
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag9():
    """
    x=2-D,offset = -1,x_shape=[3,3]
    """
    x = randtool("float", -10, 10, [3, 3])
    offset = -1
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag10():
    """
    x=2-D,offset = 10,x_shape=[3,3]
    """
    x = randtool("float", -10, 10, [3, 3])
    offset = 1
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag11():
    """
    x=1-D,offset = 0,x_shape=[3]
    """
    x = randtool("float", -10, 10, [3])
    offset = 0
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag12():
    """
    x=1-D,offset = -1,x_shape=[3]
    """
    x = randtool("float", -10, 10, [3])
    offset = -1
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag13():
    """
    x=1-D,offset = 10,x_shape=[3]
    """
    x = randtool("float", -10, 10, [3])
    offset = 10
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag14():
    """
    x=2-D,offset = 0,x_shape=[3,2]
    """
    x = randtool("float", -10, 10, [3, 2])
    offset = 0
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag15():
    """
    x=2-D,offset = -1,x_shape=[3,2]
    """
    x = randtool("float", -10, 10, [3, 2])
    offset = -1
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag16():
    """
    x=2-D,offset = 10,x_shape=[3,2]
    """
    x = randtool("float", -10, 10, [3, 2])
    offset = 1
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag17():
    """
    x=2-D,offset = 0,x_shape=[2,3]
    """
    x = randtool("float", -10, 10, [2, 3])
    offset = 0
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag18():
    """
    x=2-D,offset = -1,x_shape=[2,3]
    """
    x = randtool("float", -10, 10, [2, 3])
    offset = -1
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diag_parameters
def test_diag19():
    """
    x=2-D,offset = 10,x_shape=[2,3]
    """
    x = randtool("float", -10, 10, [2, 3])
    offset = 2
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset)


# test padding_value only on 1-D
@pytest.mark.api_base_diag_parameters
def test_diag20():
    """
    x=1-D,x_shape=[3],padding_value = 0
    """
    x = randtool("float", -10, 10, [3])
    offset = -2
    padding_value = 0
    res = np.diag(x, offset)
    obj.run(res=res, x=x, offset=offset, padding_value=padding_value)


@pytest.mark.api_base_diag_parameters
def test_diag21():
    """
    x=1-D,x_shape=[2],padding_value = 1.1
    """
    x = np.array([1, 2])
    offset = 1
    padding_value = 1.1
    res = np.array([[1.1, 1, 1.1], [1.1, 1.1, 2], [1.1, 1.1, 1.1]])
    obj.run(res=res, x=x, offset=offset, padding_value=padding_value)


@pytest.mark.api_base_diag_parameters
def test_diag22():
    """
    x=1-D,x_shape=[1],padding_value = -1.1
    """
    x = np.array([1])
    offset = -1
    padding_value = -1e-3
    res = np.array([[-1e-3, -1e-3], [1, -1e-3]])
    obj.run(res=res, x=x, offset=offset, padding_value=padding_value)


# test name
@pytest.mark.api_base_diag_parameters
def test_diag23():
    """
    x=1-D,x_shape=[2]
    """
    x = np.array([1, 2])
    name = "test name"
    res = np.diag(x)
    obj.run(res=res, x=x, name=name)


# exception case
@pytest.mark.api_base_diag_exception
def test_diag24():
    """
    x=2-D,x_shape=[3,3,3],c++error,static ValueError
    """
    x = np.zeros([3, 3, 3])
    obj.exception(mode="c", x=x, etype="InvalidArgumentError")


@pytest.mark.api_base_diag_exception
def test_diag25():
    """
    x=1-D,offset = 1e-9,x_shape=[3,1],c++error,static TypeError
    """
    x = np.array([1, 2])
    offset = 2.5
    obj.exception(mode="c", x=x, offset=offset, etype="InvalidArgumentError")


@pytest.mark.api_base_diag_exception
def test_diag26():
    """
    x=1-D,x = np.zeros([]),c++error,static ValueError
    """
    x = np.zeros([])
    obj.exception(mode="c", x=x, etype="InvalidArgumentError")
