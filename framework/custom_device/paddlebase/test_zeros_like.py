#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test zeros_like
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestZerosLike(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64, np.bool_, np.float16]
        # self.debug = True
        # self.static = True
        # enable check grad
        # 反向截断
        self.enable_backward = False


obj = TestZerosLike(paddle.zeros_like)


@pytest.mark.api_base_zeros_vartype
def test_zeros_like_base():
    """
    base
    """
    x = randtool("int", -2, 2, [3, 3, 3])
    res = np.zeros_like(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_zeros_parameters
def test_zeros_like():
    """
    default
    """
    x = randtool("int", -10, 10, [3, 3, 3])
    res = np.zeros_like(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_zeros_parameters
def test_zeros_like1():
    """
    input_type=float, large shape
    """
    x = randtool("float", -100, 100, [3, 3, 3, 3, 3, 3])
    res = np.zeros_like(x)
    obj.run(res=res, x=x)
