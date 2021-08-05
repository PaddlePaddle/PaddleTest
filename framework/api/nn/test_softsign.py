#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_softsign
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestSoftsign(APIBase):
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


obj = TestSoftsign(paddle.nn.Softsign)


@pytest.mark.api_nn_SOFTSIGN_vartype
def test_softsign_base():
    """
    base
    """
    x = np.array([-0.4, -0.2, 0.1, 0.3])
    res = x / (1 + np.abs(x))
    obj.base(res=res, data=x)


@pytest.mark.api_nn_SOFTSIGN_parameters
def test_softsign():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = x / (1 + np.abs(x))
    obj.run(res=res, data=x)
