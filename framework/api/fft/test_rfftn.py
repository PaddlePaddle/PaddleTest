#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test rfftn
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRfftn(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float64, np.float32]
        # self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestRfftn(paddle.fft.rfftn)


@pytest.mark.api_fft_rfftn_vartype
def test_rfftn_base():
    """
    base
    :return:
    """
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[(4 + 0j), (2 + 0j)], [0j, (-22 + 0j)]], [[(-2 + 0j), (4 + 0j)], [(2 + 0j), (4 + 0j)]]]
    obj.base(res=res, x=x)


@pytest.mark.api_fft_rfftn_parameters
def test_rfftn():
    """
    base
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[(4 + 0j), (2 + 0j)], [0j, (-22 + 0j)]], [[(-2 + 0j), (4 + 0j)], [(2 + 0j), (4 + 0j)]]]
    obj.run(res=res, x=x)


@pytest.mark.api_fft_rfftn_parameters
def test_rfftn1():
    """
    s = [1]
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1])


@pytest.mark.api_fft_rfftn_parameters
def test_rfftn2():
    """
    s = [1] norm="backward"
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1], norm="backward")


@pytest.mark.api_fft_rfftn_parameters
def test_rfftn3():
    """
    s = [1] norm="forward"
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1], norm="forward")


@pytest.mark.api_fft_rfftn_parameters
def test_rfftn4():
    """
    s = [1] norm="ortho"
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1], norm="ortho")


@pytest.mark.api_fft_rfftn_parameters
def test_rfftn5():
    """
    s = [1] norm="ortho" axes=[0]
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, -3.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1], axes=[0], norm="ortho")
