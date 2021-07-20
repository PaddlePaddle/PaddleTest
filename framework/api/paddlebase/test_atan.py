#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test atan
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestAtan(APIBase):
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


obj = TestAtan(paddle.atan)


@pytest.mark.api_base_atan_vartype
def test_atan_base():
    """
    base
    """
    x = -1 + 2 * np.random.random(size=[3, 3, 3])
    res = np.arctan(x)
    obj.base(res=res, x=x)
