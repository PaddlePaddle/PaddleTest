#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file test_Sigmoid.py
  * @author jiaxiao01
  * @date 2020/7/22 16:00
  * @brief test paddle.nn.Sigmoid
  *
  **************************************************************************/
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSigmoid(APIBase):
    """
    test Sigmoid
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        #self.debug = True
        #self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestSigmoid(paddle.nn.Sigmoid)

@pytest.mark.p0
def test_Sigmoid_base():
    """
    Sigmoid_base
    """
    x_data =  np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
    res = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376])
    obj.base(res=res, data=x_data)


def test_Sigmoid_input0():
    """
    input=[1.0, 2.0, 3.0, 4.0]
    """
    x_data =  np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
    res = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376])
    obj.run(res=res, data=x_data)
