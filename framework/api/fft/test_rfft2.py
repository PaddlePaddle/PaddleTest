#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
rfft2
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRfft2(APIBase):
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


obj = TestRfft2(paddle.fft.rfft2)


@pytest.mark.api_fft_rfftn_vartype
def test_rfft2_base():
    """
    base
    :return:
    """
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [
        [[1.0 + 0.0j, 3.0 + 0.0j], [1.0 + 0.0j, -9.0 + 0.0j]],
        [[3.0 + 0.0j, -1.0 + 0.0j], [-1.0 + 0.0j, -13.0 + 0.0j]],
    ]
    obj.base(res=res, x=x)


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2():
    """
    base
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [
        [[1.0 + 0.0j, 3.0 + 0.0j], [1.0 + 0.0j, -9.0 + 0.0j]],
        [[3.0 + 0.0j, -1.0 + 0.0j], [-1.0 + 0.0j, -13.0 + 0.0j]],
    ]
    obj.run(res=res, x=x)


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2_1():
    """
    s=[1, 2]
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[1.0 + 0.0j, -3.0 + 0.0j]], [[1.0 + 0.0j, -7.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1, 2])


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2_2():
    """
    s=[1, 2] norm=backward
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[1.0 + 0.0j, -3.0 + 0.0j]], [[1.0 + 0.0j, -7.0 + 0.0j]]]
    obj.run(res=res, x=x, s=[1, 2], norm="backward")


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2_3():
    """
    s=[1, 2] norm=forward
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[0.5 + 0.0j, -1.5 + 0.0j]], [[0.5 + 0.0j, -3.5 + 0.0j]]]
    obj.run(res=res, x=x, s=[1, 2], norm="forward")


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2_4():
    """
    s=[1, 2] norm=ortho
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[0.70710678 + 0.0j, -2.12132034 + 0.0j]], [[0.70710678 + 0.0j, -4.94974747 + 0.0j]]]
    obj.run(res=res, x=x, s=[1, 2], norm="ortho")


@pytest.mark.api_fft_rfftn_parameters
def test_rfft2_5():
    """
    s=[1, 2] norm=ortho axes=None
    :return:
    """
    np.random.seed(obj.seed)
    x = randtool("int", -5, 5, [2, 2, 2])
    res = [[[0.70710678 + 0.0j, -2.12132034 + 0.0j]], [[0.70710678 + 0.0j, -4.94974747 + 0.0j]]]
    obj.run(res=res, x=x, s=[1, 2], norm="ortho", axes=None)
