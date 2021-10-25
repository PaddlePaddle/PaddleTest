#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test numel
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestNumel(APIBase):
    """
    test numel
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float16, np.float32, np.float64]
        self.enable_backward = False


obj = TestNumel(paddle.numel)


@pytest.mark.api_base_numel_vartype
def test_numel_base():
    """
    base
    """
    x = np.arange(20).reshape(4, 5)
    res = np.array([20])
    obj.base(res=res, x=x)
