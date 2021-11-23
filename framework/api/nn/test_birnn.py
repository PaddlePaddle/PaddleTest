#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_birnn
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


obj = RnnBase(paddle.nn.BiRNN)
obj.enable_static = False
obj.base = False


@pytest.mark.api_nn_BiRNN_vartype
def test_birnn_base():
    """
    default
    """
    np.random.seed(22)
    x = np.random.rand(1, 1, 2)
    res = np.array([[[0.76156908, 0.76156908, 0.76156908, 0.76156908, 0.76156908, 0.76156908]]])
    cell_fw = paddle.nn.LSTMCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    cell_bw = paddle.nn.LSTMCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj.atol = 1e-4
    obj.run(res, x, cell_fw=cell_fw, cell_bw=cell_bw)


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn0():
    """
    time_major = True
    """
    np.random.seed(22)
    x = np.random.rand(2, 1, 2)
    res = np.array(
        [
            [[0.76156908, 0.76156908, 0.76156908, 0.96402740, 0.96402740, 0.96402740]],
            [[0.96402603, 0.96402603, 0.96402603, 0.76159179, 0.76159179, 0.76159179]],
        ]
    )
    cell_fw = paddle.nn.LSTMCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    cell_bw = paddle.nn.LSTMCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj.atol = 1e-5
    obj.run(res, x, cell_fw=cell_fw, cell_bw=cell_bw, time_major=True)


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn1():
    """
    set initial_states
    """
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    h_data = np.random.rand(1, 3)
    c_data = np.random.rand(1, 3)
    h1_data = np.random.rand(1, 4)
    c1_data = np.random.rand(1, 4)
    res = np.array([[[0.87139904, 0.85394186, 0.93427956, 0.94820923, 0.94081521, 0.83030295, 0.76416111]]])
    cell_fw = paddle.nn.LSTMCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    cell_bw = paddle.nn.LSTMCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj.run(res, x_data, ((h_data, c_data), (h1_data, c1_data)), cell_fw=cell_fw, cell_bw=cell_bw)


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn2():
    """
    cell: GRUCell
    set initial_states
    """
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    # h_data = np.random.rand(1, 4)
    # h1_data = np.random.rand(1, 4)
    res = np.array([[[0.00002122, 0.00002122, 0.00002122, 0.00002122, 0.00002122, 0.00002122, 0.00002122, 0.00002122]]])
    cell_fw = paddle.nn.GRUCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    cell_bw = paddle.nn.GRUCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj.atol = 1e-4
    obj.run(res, x_data, cell_fw=cell_fw, cell_bw=cell_bw)


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn3():
    """
    cell: SimpleRNNCell
    set initial_states
    """
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    h_data = np.random.rand(1, 3)
    h1_data = np.random.rand(1, 4)
    res = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]])
    cell_fw = paddle.nn.SimpleRNNCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    cell_bw = paddle.nn.SimpleRNNCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj.run(res, x_data, (h_data, h1_data), cell_fw=cell_fw, cell_bw=cell_bw)


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn4():
    """
    cell: fw -> GRUCell; bw -> SimpleRNNCell
    set initial_states
    """
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    h_data = np.random.rand(1, 3)
    h1_data = np.random.rand(1, 4)
    res = np.array([[[0.42053813, 0.85918200, 0.17116165, 1.0, 1.0, 1.0, 1.0]]])
    cell_fw = paddle.nn.GRUCell(
        2,
        3,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    cell_bw = paddle.nn.SimpleRNNCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj.run(res, x_data, (h_data, h1_data), cell_fw=cell_fw, cell_bw=cell_bw)
