#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test reciprocal
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestReciprocal(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.rtol = 0.1
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestReciprocal(paddle.reciprocal)


@pytest.mark.api_base_reciprocal_vartype
def test_reciprocal_base():
    """
    base
    """
    x = -1 + 2 * np.random.random(size=[3, 3, 3])
    res = np.reciprocal(x)
    obj.base(res=res, x=x)
