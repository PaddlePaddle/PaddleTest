#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_lstmcell
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


@pytest.mark.api_nn_LSTMCell_vartype
def test_lstmcell_base0():
    """
    test_grucell_base
    """
    obj0 = RnnBase(paddle.nn.LSTMCell)
    np.random.seed(22)
    x = np.random.rand(1, 2)
    res = np.array([[0.75616062, 0.75616062, 0.75616062]])
    obj0.atol = 1e-4
    obj0.run(
        res,
        x,
        input_size=2,
        hidden_size=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTMCell_vartype
def test_lstmcell_base1():
    """
    test_grucell_base
    """
    obj1 = RnnBase(paddle.nn.LSTMCell)
    obj1.dtype = "float64"
    np.random.seed(22)
    x = np.random.rand(1, 2)
    res = np.array([[0.75616062, 0.75616062, 0.75616062]])
    obj1.atol = 1e-4
    obj1.run(
        res,
        x,
        input_size=2,
        hidden_size=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTMCell_parameters
def test_lstmcell0():
    """
    test_grucell0
    """
    obj2 = RnnBase(paddle.nn.LSTMCell)
    obj2.enable_static = False
    obj2.atol = 1e-4
    np.random.seed(22)
    x = np.random.rand(4, 12)
    res = np.array(
        [
            [
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
                0.76159334,
            ],
            [
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
            ],
            [
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
                0.76159418,
            ],
            [
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
                0.76158971,
            ],
        ]
    )
    obj2.base = False
    obj2.run(
        res,
        x,
        input_size=12,
        hidden_size=16,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTMCell_parameters
def test_lstmcell1():
    """
    set states
    """
    obj3 = RnnBase(paddle.nn.LSTMCell)
    obj3.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2)
    h = np.random.rand(1, 4)
    c = np.random.rand(1, 4)
    res = np.array([[0.85394204, 0.93427968, 0.83977348, 0.94802976]])
    obj3.run(
        res,
        x,
        (h, c),
        input_size=2,
        hidden_size=4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
