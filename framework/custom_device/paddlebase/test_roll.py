#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_roll
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestRoll(APIBase):
    """
    test roll
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestRoll(paddle.roll)


@pytest.mark.api_base_roll_vartype
def test_roll_base():
    """
    shifts=0, axis=None
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = data
    obj.base(res=res, x=data, shifts=0, axis=None)


@pytest.mark.api_base_roll_parameters
def test_roll():
    """
    shifts=0, axis=None
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = data
    obj.run(res=res, x=data, shifts=0, axis=None)


@pytest.mark.api_base_roll_parameters
def test_roll1():
    """
    shifts=1, axis=None
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = np.array([[9, 1, 2], [3, 4, 5], [6, 7, 8]])
    obj.run(res=res, x=data, shifts=1, axis=None)


@pytest.mark.api_base_roll_parameters
def test_roll2():
    """
    shifts=-1, axis=1
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    obj.run(res=res, x=data, shifts=-1, axis=0)


@pytest.mark.api_base_roll_parameters
def test_roll3():
    """
    shifts=[-1, 1], axis=[0, 1]
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = np.array([[6, 4, 5], [9, 7, 8], [3, 1, 2]])
    obj.run(res=res, x=data, shifts=[-1, 1], axis=[0, 1])


@pytest.mark.api_base_roll_parameters
def test_roll4():
    """
    shifts=(-1, 1), axis=(0, 1)
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = np.array([[6, 4, 5], [9, 7, 8], [3, 1, 2]])
    obj.run(res=res, x=data, shifts=(-1, 1), axis=(0, 1))


@pytest.mark.api_base_roll_exception
def test_roll5():
    """
    TypeError asix is string
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # res = np.array([[6, 4, 5], [9, 7, 8], [3, 1, 2]])
    obj.exception(etype=TypeError, mode="python", x=data, shifts=(-1, 1), axis="(0, 1)")


@pytest.mark.api_base_roll_exception
def test_roll6():
    """
    ValueError asix out of bound
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # res = np.array([[6, 4, 5], [9, 7, 8], [3, 1, 2]])
    obj.exception(etype=ValueError, mode="python", x=data, shifts=(-1, 1), axis=(100))
