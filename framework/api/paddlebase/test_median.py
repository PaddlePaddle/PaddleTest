#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test median
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestMedian(APIBase):
    """
    test max
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestMedian(paddle.median)


@pytest.mark.api_base_median_vartype
def test_median_base():
    """
    test base
    Returns:

    """
    x = randtool("float", -10, 10, [4, 4, 4])
    axis = 0
    keepdims = False
    res = np.median(x, axis=axis, keepdims=keepdims)
    obj.base(res=res, x=x, axis=axis, keepdim=keepdims)


@pytest.mark.api_base_median_parameters
def test_median():
    """
    test base axis=1
    Returns:

    """
    x = randtool("float", -10, 10, [4, 4, 4])
    axis = 1
    keepdims = False
    res = np.median(x, axis=axis, keepdims=keepdims)
    obj.base(res=res, x=x, axis=axis, keepdim=keepdims)


@pytest.mark.api_base_median_parameters
def test_median1():
    """
    test base axis=1 keepdims=True
    Returns:

    """
    x = randtool("float", -10, 10, [4, 4, 4])
    axis = 1
    keepdims = True
    res = np.median(x, axis=axis, keepdims=keepdims)
    obj.base(res=res, x=x, axis=axis, keepdim=keepdims)


@pytest.mark.api_base_median_parameters
def test_median2():
    """
    test base axis=-2 keepdims=True
    Returns:

    """
    x = randtool("float", -10, 10, [4, 4, 4])
    axis = -2
    keepdims = True
    res = np.median(x, axis=axis, keepdims=keepdims)
    obj.base(res=res, x=x, axis=axis, keepdim=keepdims)
