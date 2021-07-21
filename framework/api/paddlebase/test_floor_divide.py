#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test floor divide
"""
from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestFloorDivide(APIBase):
    """
    test normal
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestFloorDivide(paddle.floor_divide)


@pytest.mark.api_base_floor_divide_vartype
def test_floor_divide_base():
    """
    base,x_shape=y_shape,x>0,y>0
    """
    x = np.array([2, 3, 4])
    y = np.array([1, 5, 2])
    res = np.floor_divide(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_floor_divide_parameters
def test_floor_divide():
    """
    x_shape>y_shape
    """
    x = randtool("int", 1, 10, [3, 3, 3])
    y = randtool("int", 1, 10, [3])
    res = np.floor_divide(x, y)
    obj.run(res=res, x=x, y=y)


# def test_floor_divide1():
#     """
#     x_shape<y_shape,revert->error
#     """
#     x = randtool('int', 1, 10, [3])
#     y = randtool('int', -1, 10, [3, 1])
#     res = np.floor_divide(x, y)
#     obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_floor_divide_parameters
def test_floor_divide2():
    """
    x=y!=0
    """
    x = np.array([-10, 9])
    y = np.array([-10, 9])
    res = np.floor_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_floor_divide_parameters
def test_floor_divide3():
    """
    x=0,y!=0
    """
    x = np.zeros([3])
    y = np.array([1, 2, 3])
    res = np.floor_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_floor_divide_parameters
def test_floor_divide4():
    """
    x<0,name = None
    """
    x = np.array([[-3], [-11], [-2]])
    y = np.array([[-1], [2], [1]])
    name = None
    res = np.array([[3], [-5], [-2]])
    obj.run(res=res, x=x, y=y, name=name)


@pytest.mark.api_base_floor_divide_parameters
def test_floor_divide5():
    """
    x_shape<y_shape, name = ""
    """
    x = np.array([[-33], [0], [32]])
    y = np.array([[4], [22], [-17]])
    name = ""
    res = np.array([[-8], [0], [-1]])
    obj.run(res=res, x=x, y=y, name=name)


# def test_floor_divide6():
#     """
#     x =y= np.array([]),[]?None,icafe
#     """
#     x = np.array([])
#     y = np.array([])
#     res = np.floor_divide(x, y)
#     obj.run(res=res, x=x, y=y)
