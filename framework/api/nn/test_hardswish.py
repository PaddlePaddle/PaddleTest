#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_hardswish
"""
import math
from apibase import APIBase
from apibase import randtool
from apibase import tanh
import paddle
import pytest
import numpy as np


class TestHardswish(APIBase):
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


obj = TestHardswish(paddle.nn.Hardswish)


@pytest.mark.api_nn_Hardswish_vartype
def test_hardswish_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    arr = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] <= -3:
            arr.append(0)
        elif x.flatten()[i] >= 3:
            arr.append(x.flatten()[i])
        else:
            arr.append(x.flatten()[i] * (x.flatten()[i] + 3) / 6)
    res = np.array(arr).reshape(x.shape)
    obj.base(res=res, data=x)


@pytest.mark.api_nn_Hardswish_parameters
def test_hardswish():
    """
     x = np.zeros((3, 4))
    """
    x = np.zeros((3, 4)).astype(np.float32)
    res = np.zeros((3, 4)).astype(np.float32)
    obj.run(res=res, data=x)
