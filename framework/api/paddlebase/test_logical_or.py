#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:exportab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file test_logical_or.py
  * @author jiaxiao01
  * @date 2020/09/25 16:00
  * @brief test paddle.logical_or
  *
  **************************************************************************/
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLogicalOr(APIBase):
    """
    test logical_or
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestLogicalOr(paddle.logical_or)


def test_logical_or_1D_tensor():
    """
    logical_or_1D_tensor
    """
    x_data = np.array([True])
    y_data = np.array([True, False, True, False])
    res = np.logical_or(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


def test_logical_or_broadcast_1():
    """
    logical_or_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.bool)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.bool)
    res = np.logical_or(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


def test_logical_or_broadcast_2():
    """
    logical_or_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.bool)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.bool)
    res = np.logical_or(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
