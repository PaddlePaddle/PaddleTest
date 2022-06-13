#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_grucell
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


@pytest.mark.api_nn_GRUCell_vartype
def test_grucell_base0():
    """
    test_grucell_base
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj0 = RnnBase(paddle.nn.GRUCell)
    np.random.seed(22)
    x = np.random.rand(1, 2)
    h = np.random.rand(1, 3)
    res = np.array([[0.42068464, 0.85921764, 0.17137116]])
    obj0.run(
        res,
        x,
        h,
        input_size=2,
        hidden_size=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_GRUCell_vartype
def test_grucell_base1():
    """
    test_grucell_base
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj1 = RnnBase(paddle.nn.GRUCell)
    obj1.dtype = "float64"
    obj1.enable_static = False
    np.random.seed(22)
    x = np.random.rand(1, 2)
    h = np.random.rand(1, 3)
    res = np.array([[0.42068464, 0.85921764, 0.17137116]])
    obj1.run(
        res,
        x,
        h,
        input_size=2,
        hidden_size=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})


@pytest.mark.api_nn_GRUCell_parameters
def test_grucell0():
    """
    test_grucell0
    """
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
    obj2 = RnnBase(paddle.nn.GRUCell)
    obj2.enable_static = False
    np.random.seed(22)
    x = np.random.rand(4, 12)
    h = np.random.rand(4, 16)
    res = np.array(
        [
            [
                0.12906402,
                0.47306001,
                0.19755852,
                0.78162277,
                0.60741389,
                0.4435001,
                0.98202908,
                0.70895541,
                0.21873045,
                0.16748828,
                0.53876549,
                0.31342077,
                0.91153699,
                0.08625567,
                0.31407404,
                0.21913737,
            ],
            [
                0.7366662,
                0.82854038,
                0.74456447,
                0.17753702,
                0.85845321,
                0.00551242,
                0.35764313,
                0.95314825,
                0.94177639,
                0.023049,
                0.67301601,
                0.16411436,
                0.56813234,
                0.24212295,
                0.15785593,
                0.50356281,
            ],
            [
                0.57179517,
                0.4699524,
                0.50816727,
                0.32375157,
                0.88649833,
                0.25183553,
                0.86208779,
                0.63519621,
                0.69722319,
                0.03045362,
                0.65731132,
                0.99932241,
                0.50654548,
                0.93264574,
                0.44080287,
                0.20002866,
            ],
            [
                0.80385357,
                0.80767208,
                0.05795288,
                0.79030603,
                0.45004165,
                0.85121077,
                0.36241686,
                0.20242763,
                0.87218946,
                0.60270798,
                0.83185273,
                0.0287326,
                0.70429665,
                0.7629959,
                0.3959372,
                0.19343626,
            ],
        ]
    )
    obj2.run(
        res,
        x,
        h,
        input_size=12,
        hidden_size=16,
        weight_ih_attr=initializer.Constant(4),
        weight_hh_attr=initializer.Constant(4),
        bias_ih_attr=initializer.Constant(4),
        bias_hh_attr=initializer.Constant(4),
    )
    paddle.fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})
