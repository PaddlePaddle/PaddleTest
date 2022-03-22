#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_calculate_gain
"""

import paddle
import pytest


@pytest.mark.api_nn_calculate_gain_parameters
def test_dirac0():
    """
    linearity function
    """
    gain = paddle.nn.initializer.calculate_gain("conv3d")
    assert gain == 1.0


@pytest.mark.api_nn_calculate_gain_parameters
def test_dirac1():
    """
    tanh
    """
    gain = paddle.nn.initializer.calculate_gain("tanh")
    assert gain == 5.0 / 3


@pytest.mark.api_nn_calculate_gain_parameters
def test_dirac2():
    """
    relu
    """
    gain = paddle.nn.initializer.calculate_gain("relu")
    assert gain == 1.4142135623730951


@pytest.mark.api_nn_calculate_gain_parameters
def test_dirac3():
    """
    sigmoid
    """
    gain = paddle.nn.initializer.calculate_gain("sigmoid")
    assert gain == 1.0


@pytest.mark.api_nn_calculate_gain_parameters
def test_dirac4():
    """
    leaky_relu
    """
    gain = paddle.nn.initializer.calculate_gain("leaky_relu")
    assert gain == 1.4141428569978354


@pytest.mark.api_nn_calculate_gain_parameters
def test_dirac5():
    """
    leaky_relu
    param=1.0
    """
    gain = paddle.nn.initializer.calculate_gain("leaky_relu", param=1.0)
    assert gain == 1.0
