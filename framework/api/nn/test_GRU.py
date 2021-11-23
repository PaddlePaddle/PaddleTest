#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_gru
"""

from rnn_base import RnnBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np


obj = RnnBase(paddle.nn.GRU)


@pytest.mark.api_nn_GRU_vartype
def test_gru_base():
    """
    Sigmoid_base
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(1, 1, 5)
    res = np.array(
        [
            [
                [0.22040676, 0.81195146, 0.01052971, 0.56120497, 0.81372672],
                [0.22041270, 0.81195289, 0.01053725, 0.56120831, 0.81372815],
            ]
        ]
    )
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru0():
    """
    default
    """
    obj.enable_static = False
    obj.base = False
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(1, 1, 5)
    res = np.array(
        [
            [
                [0.22040676, 0.81195146, 0.01052971, 0.56120497, 0.81372672],
                [0.22041270, 0.81195289, 0.01053725, 0.56120831, 0.81372815],
            ]
        ]
    )
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru1():
    """
    num_layers = 3
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(3, 1, 5)
    res = np.array(
        [
            [
                [0.70193797, 0.29757845, 0.76799279, 0.68821836, 0.38718364],
                [0.70193803, 0.29757863, 0.76799285, 0.68821841, 0.38718379],
            ]
        ]
    )
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        num_layers=3,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru2():
    """
    num_layers = 3
    direction='bidirect'
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(6, 1, 5)
    res = np.array(
        [
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ],
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ],
            ]
        ]
    )
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        num_layers=3,
        direction="bidirect",
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru3():
    """
    num_layers = 3
    direction='bidirectional'
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 4)
    h = np.random.rand(6, 1, 5)
    res = np.array(
        [
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ],
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ],
            ]
        ]
    )
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        num_layers=3,
        direction="bidirectional",
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru4():
    """
    num_layers = 3
    direction='bidirectional'
    time_major = True
    """
    np.random.seed(22)
    x = np.random.rand(2, 1, 4)
    h = np.random.rand(6, 1, 5)
    res = np.array(
        [
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ]
            ],
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ]
            ],
        ]
    )
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        num_layers=3,
        direction="bidirectional",
        time_major=True,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru4():
    """
    num_layers = 3
    direction='bidirectional'
    time_major = True
    dropout = 0.4
    """
    np.random.seed(22)
    x = np.random.rand(2, 1, 4)
    h = np.random.rand(6, 1, 5)
    res = np.array(
        [
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ]
            ],
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ]
            ],
        ]
    )
    obj.atol = 1e-4
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        num_layers=3,
        direction="bidirectional",
        time_major=True,
        dropout=0.4,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


@pytest.mark.api_nn_GRU_parameters
def test_gru4():
    """
    num_layers = 3
    direction='bidirectional'
    time_major = True
    dropout = 0.4
    """
    np.random.seed(22)
    x = np.random.rand(2, 1, 4)
    h = np.random.rand(6, 1, 5)
    res = np.array(
        [
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ]
            ],
            [
                [
                    0.92326993,
                    0.98888630,
                    0.67741078,
                    0.79516476,
                    0.02907635,
                    0.17775907,
                    0.87492776,
                    0.74493206,
                    0.50809014,
                    0.12833390,
                ]
            ],
        ]
    )
    obj.atol = 1e-3
    obj.run(
        res,
        x,
        h,
        input_size=4,
        hidden_size=5,
        num_layers=3,
        direction="bidirectional",
        time_major=True,
        dropout=0.4,
        weight_ih_attr=initializer.Constant(2),
        weight_hh_attr=initializer.Constant(2),
        bias_ih_attr=initializer.Constant(2),
        bias_hh_attr=initializer.Constant(2),
    )


# @pytest.mark.api_nn_GRU_parameters
# def test_gru5():
#     """
#     default
#     """
#     np.random.seed(22)
#     x = np.random.rand(1, 1, 4)
#     h = np.random.rand(1, 1, 5)
#     sequence_length = np.random.randint(0, 10, (1, ))
#     res = np.array([[[0.17117153, 0.33887193, 0.27054164, 0.69104505, 0.22041391]]])
#     obj.run(res, x, h, sequence_length, input_size=4, hidden_size=5, weight_ih_attr=initializer.Constant(2),
#             weight_hh_attr=initializer.Constant(2), bias_ih_attr=initializer.Constant(2),
#             bias_hh_attr=initializer.Constant(2))
