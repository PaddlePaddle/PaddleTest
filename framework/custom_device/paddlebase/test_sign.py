#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sign
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSign(APIBase):
    """
    test sign
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestSign(paddle.sign)


@pytest.mark.api_base_sign_vartype
def test_sign_base():
    """
    sign_base
    """
    x_data = np.array([3.0, 0.0, -2.0, 1.7]).astype(np.float32)
    res = np.array([1.0, 0.0, -1.0, 1.0])
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_sign_parameters
def test_sign_input0():
    """
    input=[3.0, 0.0, -2.0, 1.7]
    """
    x_data = np.array([3.0, 0.0, -2.0, 1.7]).astype(np.float32)
    res = np.array([1.0, 0.0, -1.0, 1.0])
    obj.run(res=res, x=x_data)
