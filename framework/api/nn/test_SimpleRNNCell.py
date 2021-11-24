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


@pytest.mark.api_nn_SimpleRNNCell_vartype
def test_simplernncell_base0():
    """
    test_grucell_base
    """
    obj0 = RnnBase(paddle.nn.SimpleRNNCell)
    np.random.seed(22)
    x = np.random.rand(1, 4)
    h = np.random.rand(1, 7)
    res = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    obj0.run(
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


@pytest.mark.api_nn_SimpleRNNCell_vartype
def test_simplernncell_base1():
    """
    test_grucell_base
    """
    obj1 = RnnBase(paddle.nn.SimpleRNNCell)
    obj1.dtype = "float64"
    obj1.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 4)
    h = np.random.rand(1, 7)
    res = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    obj1.run(
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
    obj2 = RnnBase(paddle.nn.SimpleRNNCell)
    obj2.enable_static = False
    np.random.seed(22)
    x = np.random.rand(4, 12)
    res = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    obj2.run(
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
    obj3 = RnnBase(paddle.nn.SimpleRNNCell)
    obj3.enable_static = False
    np.random.seed(22)
    x = np.random.rand(4, 12)
    h = np.random.rand(4, 6)
    res = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    obj3.run(
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
    obj4 = RnnBase(paddle.nn.SimpleRNNCell)
    obj4.enable_static = False
    np.random.seed(22)
    x = np.random.rand(2, 12)
    h = np.random.rand(2, 6)
    res = np.array(
        [
            [43.13631821, 43.13631821, 43.13631821, 43.13631821, 43.13631821, 43.13631821],
            [48.96537018, 48.96537018, 48.96537018, 48.96537018, 48.96537018, 48.96537018],
        ]
    )
    obj4.atol = 1e-2
    obj4.run(
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
