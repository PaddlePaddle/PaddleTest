#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_bmm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestBmm(APIBase):
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


obj = TestBmm(paddle.bmm)


@pytest.mark.api_base_bmm_vartype
def test_bmm_base():
    """
    base
    """
    x = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    y = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_bmm_parameters
def test_bmm():
    """
    default
    """
    x = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    y = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_bmm_exception
def test_bmm1():
    """
    exception x.shape != 3
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    y = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    # res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.exception(etype=ValueError, mode="python", x=x, y=y)


@pytest.mark.api_base_bmm_exception
def test_bmm2():
    """
    exception y.shape != 3
    """
    x = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    y = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    # res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.exception(etype=ValueError, mode="python", x=x, y=y)


@pytest.mark.api_base_bmm_exception
def test_bmm3():
    """
    exception x.shape != 3 and y.shape != 3
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    y = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    # res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.exception(etype=ValueError, mode="python", x=x, y=y)


@pytest.mark.api_base_bmm_exception
def test_bmm4():
    """
    exception batch x != batch y
    """
    x = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    y = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    # res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.exception(etype=ValueError, mode="python", x=x, y=y)


@pytest.mark.api_base_bmm_exception
def test_bmm5():
    """
    exception x_shape[2] != y_shape[1]
    """
    x = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    y = np.array([[[1.0, 1.0], [2.0, 2.0]], [[5.0, 5.0], [6.0, 6.0]]])
    # res = np.array([[[6.0, 6.0], [12.0, 12.0]], [[45.0, 45.0], [60.0, 60.0]]])
    obj.exception(etype=ValueError, mode="python", x=x, y=y)
