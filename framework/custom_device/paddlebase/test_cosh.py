#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test cosh
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestCosh(APIBase):
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


obj = TestCosh(paddle.cosh)


@pytest.mark.api_base_cosh_vartype
def test_cosh_base():
    """
    base
    """
    x = -1 + 2 * np.random.random(size=[3, 3, 3])
    res = (np.exp(x) + np.exp(-x)) / 2
    obj.base(res=res, x=x)
