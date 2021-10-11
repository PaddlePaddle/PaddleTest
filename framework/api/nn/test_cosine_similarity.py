#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.CosineSimilarity
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
    w12 = paddle.sum(paddle.multiply(x1, x2), axis=axis)
    w1 = paddle.sum(paddle.multiply(x1, x1), axis=axis)
    w2 = paddle.sum(paddle.multiply(x2, x2), axis=axis)
    n12 = paddle.sqrt(paddle.clip(w1 * w2, min=eps * eps))
    cos_sim = w12 / n12
    return cos_sim


@pytest.mark.api_nn_CosineSimilarity_vartype
def test_cosine_similarity_base():
    """
    base
    """
    x1 = randtool("int", -10, 10, [5, 10])
    x2 = randtool("int", -10, 10, [5, 10])

    result1 = compute_cosine_similarity(x1, x2, axis=1, eps=1e-8)
    result2 = paddle.nn.CosineSimilarity(axis=1, eps=1e-8)(x1, x2)
    compare(result1, result2)


@pytest.mark.api_nn_CosineSimilarity_parameters
def test_cosine_similarity():
    """
    default
    """
    x1 = randtool("float", -10, 10, [2, 3, 4, 4])
    x2 = randtool("float", -10, 10, [2, 3, 4, 4])

    result1 = compute_cosine_similarity(x1, x2, axis=0, eps=1e-8)
    result2 = paddle.nn.CosineSimilarity(axis=0, eps=1e-8)(x1, x2)
    compare(result1, result2)

