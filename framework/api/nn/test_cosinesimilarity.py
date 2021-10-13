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
        # self.enable_backward=False
        # self.debug = True
        # self.static = True
        # enable check grad


# obj = TestCosineSimilarity(paddle.nn.CosineSimilarity)


def cos_sim(x1, x2, axis, eps):
    """
    Cosine Similarity using numpy
    """
    w12 = np.sum(np.multiply(x1, x2), axis=axis)
    w1 = np.sum(np.multiply(x1, x1), axis=axis)
    w2 = np.sum(np.multiply(x2, x2), axis=axis)
    n12 = np.sqrt(np.clip(w1 * w2, a_min=eps * eps, a_max=None))
    return w12 / n12


@pytest.mark.api_nn_CosineSimilarity_vartype
def test_cosinesimilarity_base():
    """
    base
    """

    x1 = randtool("float", -10, 10, [1, 5])
    x2 = randtool("float", -10, 10, [1, 5])

    axis = 1
    eps = 1e-8

    res = cos_sim(x1, x2, axis, eps)

    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
    paddle_res = cos_sim_func(x1, x2).numpy()

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

    res = cos_sim(x1, x2, axis, eps)

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
        res = cos_sim(x1, x2, axis, eps)

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
        res = cos_sim(x1, x2, axis, eps)

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
    x1 = randtool("float", -100, 100, [5, 2, 3, 4])
    x2 = randtool("float", -100, 100, [2, 3, 4])

    axis = -1
    eps = 1e-8

    res = cos_sim(x1, x2, axis, eps)

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

    res = cos_sim(x1, x2, axis, eps)

    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)
    paddle_res = cos_sim_func(x1_tensor, x2_tensor).numpy()

    compare(paddle_res, res)
    # obj.run(res=res,x1=x1,x2=x2)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity6():
    """
    error axis
    axis out of range
    """
    x1 = randtool("float", -100, 100, [5, 1, 3])
    x2 = randtool("float", -100, 100, [1, 3])

    axis = 2
    eps = 1e-8

    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)

    # cos_sim_func(x1_tensor, x2_tensor)
    try:
        cos_sim_func(x1_tensor, x2_tensor)
    except Exception as e:
        # print(e)
        if "[operator < reduce_sum > error]" in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosinesimilarity7():
    """
    error type
    type is not float
    """
    x1 = randtool("int", -100, 100, [5, 1, 3])
    x2 = randtool("float", -100, 100, [5, 1, 3])

    axis = 1
    eps = 1e-8

    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)

    # cos_sim_func(x1_tensor, x2_tensor)
    try:
        cos_sim_func(x1_tensor, x2_tensor)
    except Exception as e:
        # print(e)
        if "[operator < elementwise_mul > error]" in e.args[0]:
            pass
        else:
            raise Exception


def test_cosinesimilarity8():
    """
    error shape
    Broadcast dimension mismatch
    """
    x1 = randtool("float", -100, 100, [5, 1, 3])
    x2 = randtool("float", -100, 100, [2, 3])

    axis = 1
    eps = 1e-8

    x1_tensor = paddle.to_tensor(x1)
    x2_tensor = paddle.to_tensor(x2)

    cos_sim_func = paddle.nn.CosineSimilarity(axis=axis, eps=eps)

    # cos_sim_func(x1_tensor, x2_tensor)
    try:
        cos_sim_func(x1_tensor, x2_tensor)
    except Exception as e:
        # print(e)
        if "[operator < elementwise_mul > error]" in e.args[0]:
            pass
        else:
            raise Exception
