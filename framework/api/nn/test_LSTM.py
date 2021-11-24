#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_lstm
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


@pytest.mark.api_nn_LSTM_vartype
def test_lstm_base():
    """
    test_lstm_base
    """
    obj0 = RnnBase(paddle.nn.LSTM)
    obj1 = RnnBase(paddle.nn.LSTM)
    obj1.dtype = "float64"
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    res = np.array([[[0.76117337, 0.76117337, 0.76117337], [0.96399134, 0.96399134, 0.96399134]]])
    obj0.atol = 1e-4
    obj0.run(
        res,
        x,
        input_size=4,
        hidden_size=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    obj1.atol = 1e-4
    obj1.run(
        res,
        x,
        input_size=4,
        hidden_size=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTM_parameters
def test_lstm0():
    """
    default
    """
    obj2 = RnnBase(paddle.nn.LSTM)
    obj2.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    res = np.array([[[0.76136971, 0.76136971, 0.76136971], [0.96401352, 0.96401352, 0.96401352]]])
    obj2.run(
        res,
        x,
        input_size=4,
        hidden_size=3,
        num_layers=4,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTM_parameters
def test_lstm1():
    """
    set (h,c)
    """
    obj3 = RnnBase(paddle.nn.LSTM)
    obj3.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 1, 4)
    h = np.random.rand(1, 1, 5)
    c = np.random.rand(1, 1, 5)
    res = np.array([[[0.94801611, 0.76596570, 0.91560119, 0.94819546, 0.94080150]]])
    obj3.run(
        res,
        x,
        (h, c),
        input_size=4,
        hidden_size=5,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTM_parameters
def test_lstm2():
    """
    direction='bidirect'
    """
    obj4 = RnnBase(paddle.nn.LSTM)
    obj4.enable_static = False
    obj4.atol = 1e-4
    np.random.seed(22)
    x = np.random.rand(1, 1, 4)
    res = np.array([[[0.76117337, 0.76117337, 0.76117337, 0.76117337, 0.76117337, 0.76117337]]])
    obj4.run(
        res,
        x,
        input_size=4,
        hidden_size=3,
        direction="bidirect",
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTM_parameters
def test_lstm3():
    """
    direction='bidirectional'
    """
    obj5 = RnnBase(paddle.nn.LSTM)
    obj5.enable_static = False
    obj5.atol = 1e-4
    np.random.seed(22)
    x = np.random.rand(1, 1, 4)
    res = np.array([[[0.76117337, 0.76117337, 0.76117337, 0.76117337, 0.76117337, 0.76117337]]])
    obj5.run(
        res,
        x,
        input_size=4,
        hidden_size=3,
        direction="bidirectional",
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTM_parameters
def test_lstm4():
    """
    set (h,c)
    num_layers=4
    """
    obj6 = RnnBase(paddle.nn.LSTM)
    obj6.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 1, 4)
    h = np.random.rand(4, 1, 5)
    c = np.random.rand(4, 1, 5)
    res = np.array([[[0.90795821, 0.86459506, 0.78011400, 0.89322984, 0.79901594]]])
    obj6.run(
        res,
        x,
        (h, c),
        input_size=4,
        hidden_size=5,
        num_layers=4,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_LSTM_parameters
def test_lstm5():
    """
    set (h,c)
    num_layers=4
    time_major=True
    """
    obj7 = RnnBase(paddle.nn.LSTM)
    obj7.enable_static = False
    np.random.seed(22)
    x = np.random.rand(2, 1, 4)
    h = np.random.rand(4, 1, 5)
    c = np.random.rand(4, 1, 5)
    res = np.array(
        [
            [[0.79901594, 0.77993721, 0.93348533, 0.95166963, 0.80937260]],
            [[0.97021127, 0.96708643, 0.99073184, 0.99331963, 0.97188419]],
        ]
    )
    obj7.run(
        res,
        x,
        (h, c),
        input_size=4,
        hidden_size=5,
        num_layers=4,
        time_major=True,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
