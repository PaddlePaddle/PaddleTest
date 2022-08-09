#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_rnn
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


@pytest.mark.api_nn_BiRNN_vartype
def test_birnn_base():
    """
    default
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj0 = RnnBase(paddle.nn.RNN)
    obj0.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 1, 2)
    res = np.array([[[0.76156908, 0.76156908, 0.76156908, 0.76156908]]])
    cell = paddle.nn.LSTMCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj0.atol = 1e-4
    obj0.run(res, x, cell=cell)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn0():
    """
    time_major = True
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj2 = RnnBase(paddle.nn.RNN)
    obj2.enable_static = False
    np.random.seed(22)
    x = np.random.rand(2, 1, 2)
    res = np.array(
        [[[0.76156908, 0.76156908, 0.76156908, 0.76156908]], [[0.96402603, 0.96402603, 0.96402603, 0.96402603]]]
    )
    cell = paddle.nn.LSTMCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj2.atol = 1e-4
    obj2.run(res, x, cell=cell, time_major=True)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn1():
    """
    time_major = True
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj3 = RnnBase(paddle.nn.RNN)
    obj3.enable_static = False
    np.random.seed(22)
    x = np.random.rand(2, 1, 2)
    res = np.array(
        [[[0.96402740, 0.96402740, 0.96402740, 0.96402740]], [[0.76159179, 0.76159179, 0.76159179, 0.76159179]]]
    )
    cell = paddle.nn.LSTMCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )

    obj3.run(res, x, cell=cell, is_reverse=True, time_major=True)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn2():
    """
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj4 = RnnBase(paddle.nn.RNN)
    obj4.enable_static = False
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    h_data = np.random.rand(1, 4)
    c_data = np.random.rand(1, 4)

    res = np.array([[[0.85394204, 0.93427968, 0.83977348, 0.94802976]]])
    cell = paddle.nn.LSTMCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj4.atol = 1e-5
    obj4.run(res, x_data, (h_data, c_data), cell=cell)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn3():
    """
    cell: GRUCell
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj5 = RnnBase(paddle.nn.RNN)
    obj5.enable_static = False
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    h_data = np.random.rand(1, 4)
    res = np.array([[[0.42053807, 0.85918200, 0.17116153, 0.33886397]]])
    cell = paddle.nn.GRUCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj5.run(res, x_data, h_data, cell=cell)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_BiRNN_parameters
def test_birnn4():
    """
    cell:  -> SimpleRNNCell
    set initial_states
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj6 = RnnBase(paddle.nn.RNN)
    obj6.enable_static = False
    np.random.seed(22)
    x_data = np.random.rand(1, 1, 2)
    h_data = np.random.rand(1, 4)
    res = np.array([[[1.0, 1.0, 1.0, 1.0]]])
    cell = paddle.nn.SimpleRNNCell(
        2,
        4,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    obj6.run(res, x_data, h_data, cell=cell)
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
