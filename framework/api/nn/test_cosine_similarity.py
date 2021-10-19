#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.CosineSimilarity
"""
from apibase import APIBase
from apibase import randtool, compare
import paddle
import pytest
import numpy as np


class TestCosineSimilarity(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]


def compute_cosine_similarity(x1, x2, axis=1, eps=1e-8):
    """
    compute cosine similarity using numpy
    """
    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    return w12 / n12


@pytest.mark.api_nn_CosineSimilarity_vartype
def test_cosine_similarity_base():
    """
    check int data type of input
    """
    x1 = randtool("int", -10, 10, [5, 10])
    x2 = randtool("int", -10, 10, [5, 10])
    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)
    axis = 1
    eps = 1e-8
    try:
        expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)
        result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1_tensor, x2_tensor)
        compare(result.numpy(), expected_result)
    except Exception as e:
        print(e)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity():
    """
    check different combinations of axis and eps,
    """
    x1 = randtool("float", -10, 10, [2, 3, 4, 5, 6, 7])
    x2 = randtool("float", -10, 10, [2, 3, 4, 5, 6, 7])
    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)
    axis_list = [0, 1, 2, -1, -2, -3]
    eps_list = [1e-8, 1e-7, 1e-6, 1e-5]

    for axis in axis_list:
        for eps in eps_list:
            expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)
            result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1_tensor, x2_tensor)
            compare(result.numpy(), expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity2():
    """
    check larger value range, the elements of input tensor have
    value range [-10000, 10000]
    """
    x1 = randtool("float", -10000, 10000, [5, 10])
    x2 = randtool("float", -10000, 10000, [5, 10])
    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)
    axis = 1
    eps = 1e-8
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1_tensor, x2_tensor)
    compare(result.numpy(), expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity3():
    """
    check two broadcastable input shapes
    """
    x1 = randtool("float", -10, 10, [2, 3, 4, 5, 6, 7])
    x2 = randtool("float", -10, 10, [5, 6, 7])
    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)
    axis_list = [0, 1, 2, -1, -2, -3]
    eps_list = [1e-8, 1e-7, 1e-6, 1e-5]

    for axis in axis_list:
        for eps in eps_list:
            try:
                result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1_tensor, x2_tensor)
                expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)
                compare(result.numpy(), expected_result)
            except Exception as e:
                print(e)
