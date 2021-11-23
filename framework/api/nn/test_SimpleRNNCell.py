#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_simplernncell
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


obj = RnnBase(paddle.nn.SimpleRNNCell)


@pytest.mark.api_nn_SimpleRNNCell_vartype
def test_simplernncell_base():
    """
    test_grucell_base
    """
    np.random.seed(22)
    x = np.random.rand(1, 4)
    h = np.random.rand(1, 7)
    res = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=7,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRUCell_parameters
def test_grucell0():
    """
    default
    """
    obj.base = False
    obj.enable_static = False
    np.random.seed(22)
    x = np.random.rand(4, 12)
    res = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    obj.run(
        res,
        x,
        input_size=12,
        hidden_size=4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )


@pytest.mark.api_nn_GRUCell_parameters
def test_grucell1():
    """
    set states
    """
    np.random.seed(22)
    x = np.random.rand(4, 12)
    h = np.random.rand(4, 6)
    res = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    obj.run(
        res,
        x,
        h,
        input_size=12,
        hidden_size=6,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )


@pytest.mark.api_nn_GRUCell_parameters
def test_grucell2():
    """
    set states
    activation=relu
    """
    np.random.seed(22)
    x = np.random.rand(2, 12)
    h = np.random.rand(2, 6)
    res = np.array(
        [
            [43.13631821, 43.13631821, 43.13631821, 43.13631821, 43.13631821, 43.13631821],
            [48.96537018, 48.96537018, 48.96537018, 48.96537018, 48.96537018, 48.96537018],
        ]
    )
    obj.atol = 1e-2
    obj.run(
        res,
        x,
        h,
        input_size=12,
        hidden_size=6,
        activation="relu",
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
