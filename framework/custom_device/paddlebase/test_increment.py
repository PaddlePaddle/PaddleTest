#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test increment
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestIncrement(APIBase):
    """
    test increment
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestIncrement(paddle.increment)


@pytest.mark.api_base_increment_vartype
def test_increment_base():
    """
    x = 2
    value = 1.0
    """
    x = np.array([2]).astype(np.int32)
    res = np.array([3]).astype(np.int32)
    obj.base(res=res, x=x)


@pytest.mark.api_base_increment_parameters
def test_increment1():
    """
    x = 4.5
    value = 1.0
    """
    x = np.array([4.5]).astype(np.float32)
    res = np.array([5.5]).astype(np.float32)
    obj.run(res=res, x=x)


@pytest.mark.api_base_increment_parameters
def test_increment2():
    """
    x = 4.5
    value = 2.0
    """
    x = np.array([4.5]).astype(np.float32)
    value = 2.0
    res = np.array([6.5]).astype(np.float32)
    obj.run(res=res, x=x, value=value)
