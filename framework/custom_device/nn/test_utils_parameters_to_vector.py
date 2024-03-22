#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_parameters_to_vector
"""

import paddle
import paddle.nn.initializer as initializer
import numpy as np
import pytest


@pytest.mark.api_nn_parameters_to_vector_parameters
def test_parameters_to_vector0():
    """
    linear
    """

    linear = paddle.nn.Linear(10, 20)
    vec = paddle.nn.utils.parameters_to_vector(linear.parameters())
    res = []
    for i in range(2):
        res = np.append(res, linear.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters1():
    """
    conv1d
    """
    conv1 = paddle.nn.Conv1D(2, 3, 4)
    vec = paddle.nn.utils.parameters_to_vector(conv1.parameters())
    res = []
    for i in range(2):
        res = np.append(res, conv1.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters2():
    """
    conv2d
    """
    conv2 = paddle.nn.Conv2D(2, 3, 4, 5)
    vec = paddle.nn.utils.parameters_to_vector(conv2.parameters())
    res = []
    for i in range(2):
        res = np.append(res, conv2.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters3():
    """
    conv3d
    """
    conv3 = paddle.nn.Conv3D(2, 3, 4, 5, 6)
    vec = paddle.nn.utils.parameters_to_vector(conv3.parameters())
    res = []
    for i in range(2):
        res = np.append(res, conv3.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters4():
    """
    embedding
    """
    em = paddle.nn.Embedding(10, 3)
    vec = paddle.nn.utils.parameters_to_vector(em.parameters())
    res = []
    for i in range(1):
        res = np.append(res, em.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters5():
    """
    simple rnn
    """
    s_rnn = paddle.nn.SimpleRNN(8, 16, 2)
    vec = paddle.nn.utils.parameters_to_vector(s_rnn.parameters())
    res = []
    for i in range(8):
        res = np.append(res, s_rnn.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters6():
    """
    lstm rnn
    """
    lstm = paddle.nn.LSTM(8, 16, 2)
    vec = paddle.nn.utils.parameters_to_vector(lstm.parameters())
    res = []
    for i in range(8):
        res = np.append(res, lstm.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters7():
    """
    GRU rnn
    """
    gru = paddle.nn.GRU(8, 16, 2)
    vec = paddle.nn.utils.parameters_to_vector(gru.parameters())
    res = []
    for i in range(8):
        res = np.append(res, gru.parameters()[i].numpy())
    assert np.allclose(vec, res)


@pytest.mark.api_nn_vector_to_parameters_parameters
def test_vector_to_parameters8():
    """
    Transformer
    """
    transformer = paddle.nn.Transformer(128, 2, 4, 4, 512)
    vec = paddle.nn.utils.parameters_to_vector(transformer.parameters())
    res = []
    for i in range(172):
        res = np.append(res, transformer.parameters()[i].numpy())
    assert np.allclose(vec, res)
