#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_prelu
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


def ref_prelu(x, weight):
    """
    def
    """

    x_t = x.copy()
    weight = weight.reshape(1, -1, 1, 1)
    neg_indices = x <= 0
    assert x.shape == neg_indices.shape
    x_t[neg_indices] = (x_t * weight)[neg_indices]
    return (x_t,)


class TestPRelu(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestPRelu(paddle.nn.PReLU)


@pytest.mark.api_nn_PReLU_vartype
def test_prelu_base():
    """
    base
    """
    data = randtool("float", -10, 10, [2, 3, 3, 3])
    # num_parameters = 1
    init = np.array([0.25, 0.25, 0.25])
    res = np.array(ref_prelu(data, init))[0]
    obj.base(res=res, data=data)


@pytest.mark.api_nn_PReLU_parameters
def test_prelu_float64():
    """
    base float64
    """
    obj.types = [np.float64]

    paddle.set_default_dtype("float64")
    data = randtool("float", -10, 10, [2, 3, 3, 3])
    # num_parameters = 1
    init = np.array([0.25, 0.25, 0.25])
    res = np.array(ref_prelu(data, init))[0]
    obj.run(res=res, data=data)
