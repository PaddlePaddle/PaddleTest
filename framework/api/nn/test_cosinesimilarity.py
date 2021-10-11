#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_relu
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
        self.types = [np.int8, np.float32, np.float64]
        # self.enable_backward=False
        # self.debug = True
        # self.static = True
        # enable check grad


obj = TestCosineSimilarity(paddle.nn.CosineSimilarity)


@pytest.mark.api_nn_CosineSimilarity_vartype
def test_cosinesimilarity_base():
    """
    base
    """

    x1 = randtool("float", -10, 10, [1, 5])
    x2 = randtool("float", -10, 10, [1, 5])

    axis = 1
    eps = 1e-8

    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    cos_sim = w12 / n12

    res = cos_sim
    # paddle_res = paddle.nn.functional.cosine_similarity(paddle.to_tensor(x1), paddle.to_tensor(x2))
    # res = np.maximum(0, x1)
    # print(res)

    # np.random.seed(0)
    # x1 = np.random.rand(2,3)
    # x2 = np.random.rand(2,3)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
    paddle_res = cos_sim_func(x1, x2).numpy()

    # print(result)

    # x1 = paddle.to_tensor(x1)
    # x2 = paddle.to_tensor(x2)
    # obj.base(res=res, data={'x1': x1, 'x2': x2}, axis=axis, eps=eps)
    compare(paddle_res, res)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity():
    """
    input shape
    test on 3_D input
    """
    x1 = randtool("float", -10, 10, [3, 8, 2])
    x2 = randtool("float", -10, 10, [3, 8, 2])

    axis = 1
    eps = 1e-8

    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    cos_sim = w12 / n12

    res = cos_sim

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
    paddle_res = cos_sim_func(x1, x2).numpy()

    compare(paddle_res, res)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity2():
    """
    axis (int): Dimension of vectors to compute cosine similarity. Default is 1.
    test on -1, 0, 1
    """
    x1 = randtool("float", -10, 10, [1, 5, 2])
    x2 = randtool("float", -10, 10, [1, 5, 2])

    axis_list = [-2, -1, 0, 1, 2]
    eps = 1e-8

    for axis in axis_list:
        w12 = np.sum(np.multiply(x1, x2), axis=axis)
        w1 = np.sum(np.multiply(x1, x1), axis=axis)
        w2 = np.sum(np.multiply(x2, x2), axis=axis)
        n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
        cos_sim = w12 / n12

        res = cos_sim

        x1_tensor = paddle.to_tensor(x1)
        x2_tensor = paddle.to_tensor(x2)

        cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
        paddle_res = cos_sim_func(x1_tensor, x2_tensor).numpy()

        compare(paddle_res, res)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity3():
    """
    eps(float): Small value to avoid division by zero. Default is 1e-8.
    test on 1e-6, 1e-6, 1e-7, 1e-8, 1e-9
    """
    x1 = randtool("float", -10, 10, [1, 5])
    x2 = randtool("float", -10, 10, [1, 5])

    axis = 1
    eps_list = [1e-6, 1e-6, 1e-7, 1e-8, 1e-9]

    for eps in eps_list:
        w12 = np.sum(np.multiply(x1, x2), axis=axis)
        w1 = np.sum(np.multiply(x1, x1), axis=axis)
        w2 = np.sum(np.multiply(x2, x2), axis=axis)
        n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
        cos_sim = w12 / n12

        res = cos_sim

        x1_tensor = paddle.to_tensor(x1)
        x2_tensor = paddle.to_tensor(x2)

        cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
        paddle_res = cos_sim_func(x1_tensor, x2_tensor).numpy()

        compare(paddle_res, res)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity4():
    """
    input value
    test on different shape
    """
    x1 = randtool("float", -100, 100, [5, 1, 2, 3, 4])
    x2 = randtool("float", -100, 100, [2, 3, 4])

    axis = 1
    eps = 1e-8

    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    cos_sim = w12 / n12

    res = cos_sim

    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
    paddle_res = cos_sim_func(x1_tensor, x2_tensor).numpy()

    compare(paddle_res, res)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity5():
    """
    input value
    test on different shape and axis set at 0
    """
    x1 = randtool("float", -100, 100, [5, 1, 3])
    x2 = randtool("float", -100, 100, [1, 3])

    axis = 0
    eps = 1e-8

    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    cos_sim = w12 / n12

    res = cos_sim

    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
    paddle_res = cos_sim_func(x1_tensor, x2_tensor).numpy()

    compare(paddle_res, res)
