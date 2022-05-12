#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test mode
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestMode(APIBase):
    """
    test mode
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        self.static = False
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestMode(paddle.mode)
obj1 = TestMode(paddle.mode)
obj1.static = False


@pytest.mark.api_base_mode_vartype
def test_mode_base():
    """
    base
    """
    x = np.array(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]], [[1.0, 10.0, 10.0], [1.0, 0.0, 0.0]], [[1.0, 6.0, 6.0], [1.0, 3.0, 3.0]]]
    )
    res = (np.array([[1.0, 2.0], [10.0, 0.0], [6.0, 3.0]]), np.array([[2, 2], [2, 2], [2, 2]]))
    obj.base(res=res, x=x)


@pytest.mark.api_base_mode_parameters
def test_mode():
    """
    axis = 1
    keepdim = False
    """
    axis = 1
    keepdim = False
    x = np.array(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]], [[1.0, 10.0, 10.0], [1.0, 0.0, 0.0]], [[1.0, 6.0, 6.0], [1.0, 3.0, 3.0]]]
    )
    res = (np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 3.0, 3.0]]), np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]]))
    obj.run(res=res, x=x, axis=axis, keepdim=keepdim)


@pytest.mark.api_base_mode_parameters
def test_mode1():
    """
    axis = 2
    keepdim = True
    """
    axis = 2
    keepdim = True
    x = np.array(
        [[[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]], [[1.0, 10.0, 10.0], [1.0, 0.0, 0.0]], [[1.0, 6.0, 6.0], [1.0, 3.0, 3.0]]]
    )
    res = (np.array([[[1.0], [2.0]], [[10.0], [0.0]], [[6.0], [3.0]]]), np.array([[[2], [2]], [[2], [2]], [[2], [2]]]))
    obj.run(res=res, x=x, axis=axis, keepdim=keepdim)
