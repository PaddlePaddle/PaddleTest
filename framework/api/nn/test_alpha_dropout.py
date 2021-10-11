#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
    test AlphaDropout
"""
from paddle.fluid import layers

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAlphaDropout(APIBase):
    """
    test
    """

    def hook(self):
        self.types = [np.float32, np.float64]
        self.seed = 100
        self.enable_backward = False


obj = TestAlphaDropout(paddle.nn.AlphaDropout)


def numpy_alpha_dropout(x, p, training=True):
    """
    numpy version alpha dropout
    """

    def f_scale(x, scale=1.0, bias=0.0):
        out = scale * x + bias
        return out

    if training:
        if p == 1:
            return f_scale(x, scale=0.0)
        # get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
        b = -a * alpha_p * p

        dtype = x.dtype
        input_shape = x.shape
        random_tensor = layers.uniform_random(input_shape, dtype="float32", min=0.0, max=1.0)
        random_tensor = random_tensor.numpy()
        p = np.ones(input_shape, dtype="float32") * p
        keep_mask = np.greater_equal(random_tensor, p)
        keep_mask = keep_mask.astype(dtype)
        drop_mask = np.subtract(np.ones(shape=input_shape), keep_mask)

        b = np.ones(input_shape, dtype=dtype) * b
        y = x * keep_mask + f_scale(drop_mask, scale=alpha_p)
        res = f_scale(y, scale=a) + b
        return res
    else:
        return x


@pytest.mark.api_nn_AlphaDropout_vartype
def test_alpha_dropout_base():
    """
    base
    """
    x = randtool("float", 0, 2, [2, 3])
    p = 0.5
    paddle.seed(100)
    res = numpy_alpha_dropout(x, p)
    obj.run(res=res, data=x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout1():
    """
    default
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 0.5  # defult is 0.5
    res = numpy_alpha_dropout(x, p)
    obj.run(res=res, data=x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p=1
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 1.0  # defult is 0.5
    res = numpy_alpha_dropout(x, p)
    obj.run(res=res, data=x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p=0
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 0.0  # defult is 0.5
    res = numpy_alpha_dropout(x, p)
    obj.run(res=res, data=x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p = -1
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=ValueError, mode="python", data=x, p=-1)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p = 2
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=ValueError, mode="python", data=x, p=-2)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p = '1'
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=TypeError, mode="python", data=x, p="1")
