#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_hardswish
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalHardswish(APIBase):
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
        # self.delta = 1e-1


obj = TestFunctionalHardswish(paddle.nn.functional.hardswish)


@pytest.mark.api_nn_hardswish_vartype
def test_hardswish_base():
    """
    base
    """
    # x = np.array([-100, 0.3, 2.5, 100, 99, 88])
    x = randtool("float", -100, 100, [2, 4])
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] <= -3:
            res.append(0)
        elif x.flatten()[i] >= 3:
            res.append(x.flatten()[i])
        else:
            res.append(x.flatten()[i] * (3 + x.flatten()[i]) / 6)
    res = np.array(res).reshape(x.shape)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_hardswish_parameters
def test_hardswish():
    """
    default
    """
    x = randtool("float", -10, 10, [4, 2, 4])
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] <= -3:
            res.append(0)
        elif x.flatten()[i] >= 3:
            res.append(x.flatten()[i])
        else:
            res.append(x.flatten()[i] * (3 + x.flatten()[i]) / 6)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x)
