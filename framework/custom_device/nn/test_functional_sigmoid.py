#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_sigmoid
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFunctionalSigmoid(APIBase):
    """
    test sigmoid
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestFunctionalSigmoid(paddle.nn.functional.sigmoid)


@pytest.mark.api_nn_sigmoid_vartype
def test_sigmoid_base():
    """
    Sigmoid_base
    """
    x_data = np.array([1.0, 2.0, 3.0, 4.0]).astype("float32")
    res = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376])
    obj.base(res=res, x=x_data)


@pytest.mark.api_nn_sigmoid_parameters
def test_sigmoid_input0():
    """
    input=[1.0, 2.0, 3.0, 4.0]
    """
    x_data = np.array([1.0, 2.0, 3.0, 4.0]).astype("float32")
    res = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376])
    obj.run(res=res, x=x_data)
