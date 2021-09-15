#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test T
"""
from paddle.fluid.layers.nn import shape
from apibase import compare
import paddle
import pytest
import numpy as np


@pytest.mark.api_base_T_parameters
def test_T():
    """
    test base function
    """
    x = np.random.random(size=[3, 5, 4])
    paddle_x = paddle.to_tensor(x)
    compare(x.T, paddle_x.T)


@pytest.mark.api_base_T_parameters
def test_T1():
    """
    test bigger size
    """
    x = np.random.random(size=[3, 5, 4, 2, 6])
    paddle_x = paddle.to_tensor(x)
    compare(x.T, paddle_x.T)


@pytest.mark.api_base_T_parameters
def test_T2():
    """
    test T.T
    """
    x = np.random.random(size=[3, 5, 4, 2, 6])
    paddle_x = paddle.to_tensor(x)
    compare(x.T.T, paddle_x.T.T)


@pytest.mark.api_base_T_parameters
def test_T3():
    """
    test T.T.T
    """
    x = np.random.random(size=[3, 5, 4, 2, 6])
    paddle_x = paddle.to_tensor(x)
    compare(x.T.T.T, paddle_x.T.T.T)
