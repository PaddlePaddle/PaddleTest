#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_cosine_similarity
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFunctionalCosSim(APIBase):
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
        self.enable_backward = True


obj = TestFunctionalCosSim(paddle.nn.functional.cosine_similarity)


def np_cos_sim(x1, x2, dim=1, eps=1e-8):
    """
    numpy's cos_sim function
    """
    w12 = np.sum(x1 * x2, axis=dim)
    w1 = np.sum(x1 * x1, axis=dim)
    w2 = np.sum(x2 * x2, axis=dim)
    n12 = np.sqrt(np.clip(w1 * w2, eps * eps, None))
    cos_sim = w12 / n12
    return cos_sim


@pytest.mark.api_nn_cosine_similarity_vartype
def test_cossim_base():
    """
    base
    """
    np.random.seed(0)
    x1 = np.random.rand(2, 3)
    x2 = np.random.rand(2, 3)
    res = np_cos_sim(x1, x2)
    obj.base(res=res, x1=x1, x2=x2)


@pytest.mark.api_nn_cosine_similarity_parameters
def test_cossim1():
    """
    base axis = 0, dim=axis=0
    """
    np.random.seed(0)
    x1 = np.random.rand(2, 3)
    x2 = np.random.rand(2, 3)
    res = np_cos_sim(x1, x2, dim=0)
    obj.run(res=res, x1=x1, x2=x2, axis=0)


@pytest.mark.api_nn_cosine_similarity_parameters
def test_cossim2():
    """
    base axis = 0, dim=axis=1
    """
    np.random.seed(0)
    x1 = np.random.rand(2, 3)
    x2 = np.random.rand(2, 3)
    res = np_cos_sim(x1, x2, dim=1)
    obj.run(res=res, x1=x1, x2=x2, axis=1)


@pytest.mark.api_nn_cosine_similarity_parameters
def test_cossim3():
    """
    base axis = 0, dim=axis=1, eps=0
    """
    np.random.seed(0)
    x1 = np.random.rand(2, 3)
    x2 = np.random.rand(2, 3)
    res = np_cos_sim(x1, x2, dim=1, eps=0)
    obj.run(res=res, x1=x1, x2=x2, axis=1, eps=0)
