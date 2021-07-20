#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test acos
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAcos(APIBase):
    """
    test acos
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.rtol = 0.1
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestAcos(paddle.acos)


@pytest.mark.api_base_abs_vartype
def test_acos_base():
    """
    base
    """
    x = randtool("float", -1, 1, (3, 3, 3))
    res = np.arccos(x)
    obj.base(res=res, x=x)
