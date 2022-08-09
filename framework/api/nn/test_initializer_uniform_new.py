#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer_uniform
"""
from apibase import APIBase
from apibase import compare
import pytest
import paddle
import numpy as np


@pytest.mark.api_initializer_uniform_vartype
def test_initializer_uniform_base():
    """
    base
    """
    w_init = paddle.nn.initializer.Uniform()
    Linear = paddle.nn.Linear(10000, 10000, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    assert (np.maximum(w, np.ones((10000, 10000)) * 1.0) == np.ones((10000, 10000)) * 1.0).all()
    assert (np.minimum(w, np.ones((10000, 10000)) * -1.0) == np.ones((10000, 10000)) * -1.0).all()
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(10000), delta=1e-1, rtol=1e-1)
    compare(np.mean(w, axis=1), np.zeros(10000), delta=1e-1, rtol=1e-1)


@pytest.mark.api_initializer_uniform_parameters
def test_initializer_uniform1():
    """
    base
    """
    w_init = paddle.nn.initializer.Uniform(low=-2.0, high=5.0)
    Linear = paddle.nn.Linear(10000, 10000, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    assert (np.maximum(w, np.ones((10000, 10000)) * 5.0) == np.ones((10000, 10000)) * 5.0).all()
    assert (np.minimum(w, np.ones((10000, 10000)) * -2.0) == np.ones((10000, 10000)) * -2.0).all()
    compare(np.array(np.mean(w)), 1.5, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.ones(10000) * 1.5, delta=1e-1, rtol=1e-1)
    compare(np.mean(w, axis=1), np.ones(10000) * 1.5, delta=1e-1, rtol=1e-1)
