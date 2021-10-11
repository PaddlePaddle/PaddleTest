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
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestCosineSimilarity(paddle.nn.CosineSimilarity)


def compute_cosine_similarity(x1, x2, axis=1, eps=1e-8):
    """
    compute cosine similarity
    """
    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    cos_sim = w12 / n12
    return cos_sim


@pytest.mark.api_nn_CosineSimilarity_vartype
def test_cosine_similarity_base():
    """
    base
    """
    x1 = randtool("int", -10, 10, [5, 10])
    x2 = randtool("int", -10, 10, [5, 10])
    axis = 1
    eps = 1e-8
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1, x2)

    compare(result, expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity():
    """
    default
    """
    x1 = randtool("float", -10, 10, [5, 10])
    x2 = randtool("float", -10, 10, [5, 10])
    axis = 1
    eps = 1e-8
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1, x2)

    compare(result, expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity2():
    """
    modify value range
    """
    x1 = randtool("float", -10000, 10000, [5, 10])
    x2 = randtool("float", -10000, 10000, [5, 10])
    axis = 1
    eps = 1e-8
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1, x2)

    compare(result, expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity3():
    """
    modify input shape
    """
    x1 = randtool("float", -10, 10, [5, 6, 7, 8, 10])
    x2 = randtool("float", -10, 10, [5, 6, 7, 8, 10])
    axis = 1
    eps = 1e-8
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1, x2)

    compare(result, expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity4():
    """
    modify axis
    """
    x1 = randtool("float", -10, 10, [5, 10])
    x2 = randtool("float", -10, 10, [5, 10])
    axis = 0
    eps = 1e-8
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1, x2)

    compare(result, expected_result)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity5():
    """
    modify eps
    """
    x1 = randtool("float", -10, 10, [5, 10])
    x2 = randtool("float", -10, 10, [5, 10])
    axis = 1
    eps = 1e-4
    expected_result = compute_cosine_similarity(x1, x2, axis=axis, eps=eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.CosineSimilarity(axis=axis, eps=eps)(x1, x2)

    compare(result, expected_result)
