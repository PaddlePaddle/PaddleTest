#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_crop
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCrop(APIBase):
    """
    test crop
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        self.enable_backward = False
        self.no_grad_var = ["shape"]


obj = TestCrop(paddle.crop)


@pytest.mark.api_base_crop_vartype
def test_crop_base():
    """
    base
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    shape = [2, 2]
    res = np.array([[1, 2], [4, 5]])
    obj.base(res=res, x=x, shape=shape)


@pytest.mark.api_base_gather_parameters
def test_crop1():
    """
    shape = [2, 2]
    offsets = (1, 1)
    res = np.array([[5,6], [8,9]])
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    shape = [2, 2]
    offsets = (1, 1)
    res = np.array([[5, 6], [8, 9]])
    obj.run(res=res, x=x, shape=shape, offsets=offsets)


@pytest.mark.api_base_gather_parameters
def test_crop2():
    """
    shape = [2, 2]
    offsets = (0, 0)
    res = np.array([[1,2], [4,5]])
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    shape = [2, 2]
    offsets = (0, 0)
    res = np.array([[1, 2], [4, 5]])
    obj.run(res=res, x=x, shape=shape, offsets=offsets)


@pytest.mark.api_base_gather_parameters
def test_crop3():
    """
    shape = [2, 2]
    offsets = (0, 1)
    res = np.array([[2,3], [5,6]])
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    shape = [2, 2]
    offsets = (0, 1)
    res = np.array([[2, 3], [5, 6]])
    obj.run(res=res, x=x, shape=shape, offsets=offsets)


@pytest.mark.api_base_gather_parameters
def test_crop4():
    """
    shape = [2, 2]
    offsets = [1, 0]
    res = np.array([[5,6], [8,9]])
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    shape = [2, 2]
    offsets = [1, 0]
    res = np.array([[4, 5], [7, 8]])
    obj.run(res=res, x=x, shape=shape, offsets=offsets)


@pytest.mark.api_base_gather_parameters
def test_crop5():
    """
    shape = [2, 2]
    offsets = [1, 0]
    res = np.array([[5,6], [8,9]])
    """
    x = randtool("float", -10, 10, [2, 3, 3, 3])
    shape = [2, 1, -1, 2]
    offsets = [0, 0, 1, 1]
    res = np.array(
        [
            [[[7.40791377, -6.29920146], [9.06504063, 3.60901609]]],
            [[[6.43138915, 0.55102135], [-2.92035609, -8.41939397]]],
        ]
    )
    obj.run(res=res, x=x, shape=shape, offsets=offsets)
