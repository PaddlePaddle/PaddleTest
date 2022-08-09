#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_vector_to_parameters
"""

import paddle
import paddle.nn.initializer as initializer
import numpy as np
import pytest


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters0():
    """
    linear
    """
    weight_attr = paddle.ParamAttr(initializer=initializer.Constant(3.0))
    bias_attr = paddle.ParamAttr(initializer=initializer.Constant(4.0))
    linear1 = paddle.nn.Linear(2, 2, weight_attr, bias_attr)
    vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())
    linear2 = paddle.nn.Linear(2, 2)
    paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())
    for i in range(2):
        assert np.allclose(linear1.parameters()[i].numpy(), linear2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters1():
    """
    conv1d
    """
    conv1 = paddle.nn.Conv1D(2, 3, 4)
    vec = paddle.nn.utils.parameters_to_vector(conv1.parameters())
    conv2 = paddle.nn.Conv1D(2, 3, 4)
    paddle.nn.utils.vector_to_parameters(vec, conv2.parameters())
    for i in range(2):
        assert np.allclose(conv1.parameters()[i].numpy(), conv2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters2():
    """
    conv2d
    """
    conv1 = paddle.nn.Conv2D(2, 3, 4, 5)
    vec = paddle.nn.utils.parameters_to_vector(conv1.parameters())
    conv2 = paddle.nn.Conv2D(2, 3, 4, 5)
    paddle.nn.utils.vector_to_parameters(vec, conv2.parameters())
    for i in range(2):
        assert np.allclose(conv1.parameters()[i].numpy(), conv2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters3():
    """
    conv3d
    """
    conv1 = paddle.nn.Conv3D(2, 3, 4, 5, 6)
    vec = paddle.nn.utils.parameters_to_vector(conv1.parameters())
    conv2 = paddle.nn.Conv3D(2, 3, 4, 5, 6)
    paddle.nn.utils.vector_to_parameters(vec, conv2.parameters())
    for i in range(2):
        assert np.allclose(conv1.parameters()[i].numpy(), conv2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters4():
    """
    embedding
    """
    em1 = paddle.nn.Embedding(10, 3)
    vec = paddle.nn.utils.parameters_to_vector(em1.parameters())
    em2 = paddle.nn.Embedding(10, 3)
    paddle.nn.utils.vector_to_parameters(vec, em2.parameters())
    for i in range(1):
        assert np.allclose(em1.parameters()[i].numpy(), em2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters5():
    """
    simple rnn
    """
    s_rnn1 = paddle.nn.SimpleRNN(8, 16, 2)
    vec = paddle.nn.utils.parameters_to_vector(s_rnn1.parameters())
    s_rnn2 = paddle.nn.SimpleRNN(8, 16, 2)
    paddle.nn.utils.vector_to_parameters(vec, s_rnn2.parameters())
    for i in range(8):
        assert np.allclose(s_rnn1.parameters()[i].numpy(), s_rnn2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters6():
    """
    lstm rnn
    """
    lstm1 = paddle.nn.LSTM(8, 16, 2)
    vec = paddle.nn.utils.parameters_to_vector(lstm1.parameters())
    lstm2 = paddle.nn.LSTM(8, 16, 2)
    paddle.nn.utils.vector_to_parameters(vec, lstm2.parameters())
    for i in range(8):
        assert np.allclose(lstm1.parameters()[i].numpy(), lstm2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters7():
    """
    GRU rnn
    """
    gru1 = paddle.nn.GRU(8, 16, 2)
    vec = paddle.nn.utils.parameters_to_vector(gru1.parameters())
    gru2 = paddle.nn.GRU(8, 16, 2)
    paddle.nn.utils.vector_to_parameters(vec, gru2.parameters())
    for i in range(8):
        assert np.allclose(gru1.parameters()[i].numpy(), gru2.parameters()[i].numpy())


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters8():
    """
    Transformer
    """
    transformer1 = paddle.nn.Transformer(128, 2, 4, 4, 512)
    vec = paddle.nn.utils.parameters_to_vector(transformer1.parameters())
    transformer2 = paddle.nn.Transformer(128, 2, 4, 4, 512)
    paddle.nn.utils.vector_to_parameters(vec, transformer2.parameters())
    for i in range(172):
        assert np.allclose(transformer1.parameters()[i].numpy(), transformer2.parameters()[i].numpy())
