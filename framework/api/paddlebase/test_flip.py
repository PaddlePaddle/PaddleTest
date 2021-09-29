#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_flip.py
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFlip(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestFlip(paddle.flip)


@pytest.mark.api_base_flip_parameters
def test_flip_base():
    """
    base
    """
    x = randtool("int", -10, 10, [3, 3, 3])
    axis = [0]
    res = np.flip(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_flip_parameters
def test_flip():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = [0]
    res = np.flip(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_flip_vartype
def test_flip1():
    """
    types=np.bool
    """
    obj.types = [np.bool]
    x = randtool("int", -10, 10, [3, 3, 3]).astype(np.bool)
    axis = [0]
    res = np.flip(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)
    obj.types = [np.int32, np.int64, np.float32, np.float64]


@pytest.mark.api_base_flip_parameters
def test_flip2():
    """
    axis=[0,1,2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = [0, 1, 2]
    res = np.flip(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_flip_parameters
def test_flip3():
    """
    axis=[-1, 0, 1]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = [-1, 0, 1]
    res = np.flip(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_flip_parameters
def test_flip4():
    """
    axis=[-1, 0, 3, 4, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3, 3, 3, 3])
    axis = [-1, 0, 3, 4, 2]
    res = np.flip(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_flip_exception
def test_flip5():
    """
    axis=[0,1,2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = [0, 1, 4]
    # res = [123]
    obj.exception(etype="InvalidArgument", x=x, axis=axis)
