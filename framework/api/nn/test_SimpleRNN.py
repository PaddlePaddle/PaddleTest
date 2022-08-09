#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_simplernn
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np

paddle.seed(22)


@pytest.mark.api_nn_SimpleRNN_vartype
def test_simplernn_base0():
    """
    test_grucell_base
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj0 = RnnBase(paddle.nn.SimpleRNN)
    np.random.seed(22)
    x = np.random.rand(1, 2, 3)
    res = np.array([[[0.99999225, 0.99999225], [1.0, 1.0]]])
    obj0.run(
        res,
        x,
        input_size=3,
        hidden_size=2,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_vartype
def test_simplernn_base1():
    """
    test_grucell_base
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj1 = RnnBase(paddle.nn.SimpleRNN)
    obj1.dtype = "float64"
    obj1.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 3)
    res = np.array([[[0.99999225, 0.99999225], [1.0, 1.0]]])
    obj1.run(
        res,
        x,
        input_size=3,
        hidden_size=2,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_parameters
def test_simplernn0():
    """
    default
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj2 = RnnBase(paddle.nn.SimpleRNN)
    obj2.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 3)
    h = np.random.rand(1, 1, 4)
    res = np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    obj2.run(
        res,
        x,
        h,
        input_size=3,
        hidden_size=4,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_parameters
def test_simplernn1():
    """
    num_layer=2
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj3 = RnnBase(paddle.nn.SimpleRNN)
    obj3.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(2, 1, 4)
    res = np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    obj3.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=4,
        num_layers=2,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_parameters
def test_simplernn2():
    """
    num_layer=2
    activation='relu'
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj4 = RnnBase(paddle.nn.SimpleRNN)
    obj4.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(2, 1, 4)
    res = np.array(
        [
            [
                [96.69131470, 96.69131470, 96.69131470, 96.69131470],
                [1546.54138184, 1546.54138184, 1546.54138184, 1546.54138184],
            ]
        ]
    )
    obj4.atol = 1e-2
    obj4.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=4,
        num_layers=2,
        activation="relu",
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_parameters
def test_simplernn3():
    """
    time_major=True
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj5 = RnnBase(paddle.nn.SimpleRNN)
    obj5.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 1, 4)
    h = np.random.rand(1, 1, 5)
    res = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0]]])
    obj5.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        time_major=True,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_parameters
def test_simplernn4():
    """
    dropout=0.8
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj6 = RnnBase(paddle.nn.SimpleRNN)
    obj6.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(1, 1, 4)
    res = np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    obj6.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=4,
        dropout=0.8,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_SimpleRNN_parameters
def test_simplernn5():
    """
    dropout=0.8
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj7 = RnnBase(paddle.nn.SimpleRNN)
    obj7.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(1, 1, 4)
    ls = np.random.randint(2, 4, (1,))
    res = np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    obj7.run(
        res,
        x,
        h,
        ls,
        input_size=4,
        hidden_size=4,
        dropout=0.8,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
