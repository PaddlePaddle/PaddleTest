#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ceil
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestCeil(APIBase):
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


obj = TestCeil(paddle.ceil)


@pytest.mark.api_base_ceil_vartype
def test_ceil_base():
    """
    base
    """
    x = -1 + 2 * np.random.random(size=[3, 3, 3])
    res = np.ceil(x)
    obj.base(res=res, x=x)
