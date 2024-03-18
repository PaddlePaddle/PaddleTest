#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_hardsigmoid
"""

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestHardsigmoid(APIBase):
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
        self.delta = 1e-1
        # self.enable_backward = True


obj = TestHardsigmoid(paddle.nn.Hardsigmoid)


@pytest.mark.api_nn_Hardsigmoid_vartype
def test_hardsigmoid_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 2])
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] <= -3:
            res.append(0)
        elif x.flatten()[i] >= 3:
            res.append(1)
        else:
            res.append(x.flatten()[i] / 6 + 0.5)
    res = np.array(res).reshape(x.shape)
    # print(res)
    obj.base(res=res, data=x)


@pytest.mark.api_nn_Hardsigmoid_parameters
def test_hardsigmoid():
    """
    x = [[3, 3, 3], [-5, 0, 5], [-3, -3, -3]]
    """
    x = np.array([[3, 3, 3], [-5, 0, 5], [-3, -3, -3]]).astype(np.float32)
    # print(x)
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] <= -3:
            res.append(0)
        elif x.flatten()[i] >= 3:
            res.append(1)
        else:
            res.append(x.flatten()[i] / 6 + 0.5)
    res = np.array(res).reshape(x.shape)
    # print(res)
    obj.run(res=res, data=x)
