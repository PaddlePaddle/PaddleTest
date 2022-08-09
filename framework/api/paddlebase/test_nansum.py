#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test nansum
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestNansum(APIBase):
    """
    test nansum
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestNansum(paddle.nansum)


@pytest.mark.api_base_nansum_vartype
def test_nansum_base():
    """
    base
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    x = x.astype("float32")
    res = [np.nansum(x)]
    obj.run(res=res, x=x)


@pytest.mark.api_base_nansum_parameters
def test_nansum():
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float('nan')
    x[:, 1, :] = float('-nan')
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    x = x.astype("float64")
    res = np.nansum(x, axis=0)
    obj.run(res=res, x=x, axis=0)


@pytest.mark.api_base_nansum_parameters
def test_nansum1():
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float('nan')
    x[:, 1, :] = float('-nan')
    res = np.nansum(x, axis=-1)
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    res = np.nansum(x, axis=-1)
    obj.run(res=res, x=x, axis=-1)


@pytest.mark.api_base_nansum_parameters
def test_nansum2():
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float('nan')
    x[:, 1, :] = float('-nan')
    res = np.nansum(x, axis=0, keepdim=True)
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    x[-1, :, :] = float("nan")
    x[:, 1, :] = float("-nan")
    res = np.nansum(x, axis=0, keepdims=True)
    obj.run(res=res, x=x, axis=0, keepdim=True)


@pytest.mark.api_base_nansum_parameters
def test_nansum3():
    """
    x=0
    """
    x = randtool("float", -10, 10, (3, 2, 3, 4, 5, 1, 2))
    x[-1, :, 2, 2, :, :, :] = float("nan")
    x[:, 1, :, :, :, :, -1] = float("-nan")
    res = np.nansum(x, axis=3, keepdims=True)
    obj.run(res=res, x=x, axis=3, keepdim=True)


# x = randtool("float", -10, 10, (3, 3, 3))
# x[-1, :, :] = float('nan')
# x[:, 1, :] = float('-nan')
# # res = np.nansum(x.astype('float32'))
# res = np.array(np.nansum(x))
# print(res)
