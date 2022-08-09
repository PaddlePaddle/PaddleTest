#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_histogram
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestHistogram(APIBase):
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
        # self.delta = 1e-2


obj = TestHistogram(paddle.histogram)


def cal_histogram(x, bins=100, min=0, max=0):
    """
    calculate histogram api
    """
    if min == max == 0:
        min = x.min()
        max = x.max()

    gap = (max - min) / bins
    res = [0] * bins
    for i in x.flatten():
        for j in range(bins):
            if i == max:
                res[-1] += 1
                break
            if min + j * gap <= i < min + (j + 1) * gap:
                res[j] += 1
                break
    return res


@pytest.mark.api_linalg_histogram_vartype
def test_histogram_base():
    """
    base
    default
    """
    x = randtool("int", 0, 100, [4, 4])
    # x = np.random.rand(4, 4) * 100
    res = cal_histogram(x)
    obj.base(res=res, input=x)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram0():
    """
    default
    """
    x = randtool("float", 0, 2, [4, 4])
    res = cal_histogram(x)
    obj.run(res=res, input=x)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram1():
    """
    input: vector
    """
    x = randtool("float", 0, 2, [44])
    res = cal_histogram(x)
    obj.run(res=res, input=x)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram2():
    """
    bins = 4
    """
    x = randtool("float", 0, 2, [4, 4])
    res = cal_histogram(x, bins=4)
    obj.run(res=res, input=x, bins=4)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram3():
    """
    min < 0
    """
    x = randtool("float", -4, 2, [4, 4])
    res = cal_histogram(x, bins=4, min=-4)
    obj.run(res=res, input=x, bins=4, min=-4)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram4():
    """
    min < min(x)
    """
    x = randtool("float", -4, 2, [4, 4])
    res = cal_histogram(x, bins=4, min=-40)
    obj.run(res=res, input=x, bins=4, min=-40)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram5():
    """
    min > min(x)
    """
    x = randtool("float", -4, 2, [4, 4])
    res = cal_histogram(x, bins=4, min=-2)
    obj.run(res=res, input=x, bins=4, min=-2)


@pytest.mark.api_linalg_histogram_exception
def test_histogram6():
    """
    min > max
    max -->> default
    """
    x = randtool("float", -4, 2, [4, 4])
    # res = cal_hisgogram(x, bins=4, min=0)
    obj.exception("InvalidArgument", input=x, bins=4, min=1)


@pytest.mark.api_linalg_histogram_exception
def test_histogram7():
    """
    min > max
    min -->> default
    """
    x = randtool("float", -4, 2, [4, 4])
    # res = cal_hisgogram(x, bins=4, min=0)
    obj.exception("InvalidArgument", input=x, bins=4, max=-1)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram8():
    """
    max > max(x)
    """
    x = randtool("float", -4, 2, [4, 4])
    res = cal_histogram(x, bins=4, max=5)
    obj.run(res=res, input=x, bins=4, max=5)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram9():
    """
    max < max(x)
    """
    x = randtool("float", -4, 20, [4, 4])
    res = cal_histogram(x, bins=4, max=5)
    obj.run(res=res, input=x, bins=4, max=5)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram10():
    """
    max(x) < min < max
    """
    x = randtool("float", -4, 2, [4, 4])
    res = cal_histogram(x, bins=4, min=4, max=5)
    obj.run(res=res, input=x, bins=4, min=4, max=5)


@pytest.mark.api_linalg_histogram_parameters
def test_histogram11():
    """
    min < max < min(x)
    """
    x = randtool("float", -4, 2, [4, 4])
    res = cal_histogram(x, bins=4, min=-41, max=-5)
    obj.run(res=res, input=x, bins=4, min=-41, max=-5)


@pytest.mark.api_linalg_histogram_exception
def test_histogram12():
    """
    bins < 0
    """
    x = randtool("float", -4, 2, [4, 4])
    # res = cal_hisgogram(x, bins=4, min=0)
    # New Eager mode raise ValueError instead of IndexError
    obj.exception((IndexError, ValueError), mode="python", input=x, bins=-4, max=-1)
