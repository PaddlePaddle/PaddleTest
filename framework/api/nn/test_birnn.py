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


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn0():
    """
    time_major = True
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj2 = RnnBase(paddle.nn.BiRNN)
    obj2.enable_static = False
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
    obj2.atol = 1e-5
    obj2.run(res, x, cell_fw=cell_fw, cell_bw=cell_bw, time_major=True)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn1():
    """
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj3 = RnnBase(paddle.nn.BiRNN)
    obj3.enable_static = False
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
    obj3.run(res, x_data, ((h_data, c_data), (h1_data, c1_data)), cell_fw=cell_fw, cell_bw=cell_bw)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn2():
    """
    cell: GRUCell
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj4 = RnnBase(paddle.nn.BiRNN)
    obj4.enable_static = False
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
    obj4.atol = 1e-4
    obj4.run(res, x_data, cell_fw=cell_fw, cell_bw=cell_bw)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn3():
    """
    cell: SimpleRNNCell
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj5 = RnnBase(paddle.nn.BiRNN)
    obj5.enable_static = False
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
    obj5.run(res, x_data, (h_data, h1_data), cell_fw=cell_fw, cell_bw=cell_bw)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn4():
    """
    cell: fw -> GRUCell; bw -> SimpleRNNCell
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj6 = RnnBase(paddle.nn.BiRNN)
    obj6.enable_static = False
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
    obj6.run(res, x_data, (h_data, h1_data), cell_fw=cell_fw, cell_bw=cell_bw)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
