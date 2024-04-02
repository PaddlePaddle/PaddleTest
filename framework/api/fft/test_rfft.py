#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
rfft
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRfft(APIBase):
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


obj = TestRfft(paddle.fft.rfft)


@pytest.mark.api_fft_rfftn_vartype
def test_rfft_base():
    """
    base
    :return:
    """
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[1.0 + 0.0j, -3.0 + 0.0j], [0.0 + 0.0j, 6.0 + 0.0j]], [[1.0 + 0.0j, -7.0 + 0.0j], [2.0 + 0.0j, 6.0 + 0.0j]]]
    obj.base(res=res, x=x)


@pytest.mark.api_fft_rfftn_parameters
def test_rfft():
    """
    base
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[1.0 + 0.0j, -3.0 + 0.0j], [0.0 + 0.0j, 6.0 + 0.0j]], [[1.0 + 0.0j, -7.0 + 0.0j], [2.0 + 0.0j, 6.0 + 0.0j]]]
    obj.run(res=res, x=x)


@pytest.mark.api_fft_rfftn_parameters
def test_rfft1():
    """
    n = 1
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, n=1)


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2():
    """
    n = 1 norm = backward
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, n=1, norm="backward")


@pytest.mark.api_fft_rfftn_parameters
def test_rfft3():
    """
    n = 1 norm = forward
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, n=1, norm="forward")


@pytest.mark.api_fft_rfftn_parameters
def test_rfft4():
    """
    n = 1 norm = ortho
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, n=1, norm="ortho")


@pytest.mark.api_fft_rfftn_parameters
def test_rfft5():
    """
    n = 1 norm = ortho axis=2
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[-1.0 + 0.0j], [3.0 + 0.0j]], [[-3.0 + 0.0j], [4.0 + 0.0j]]]
    obj.run(res=res, x=x, n=1, norm="ortho", axis=2)
