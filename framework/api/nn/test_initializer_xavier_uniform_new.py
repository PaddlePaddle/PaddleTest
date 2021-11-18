#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer_xavier_uniform
"""
from apibase import APIBase
from apibase import compare
import pytest
import paddle
import numpy as np


@pytest.mark.api_initializer_xavier_uniform_vartype
def test_initializer_xavier_uniform_base():
    """
    base
    """
    n1 = 15000
    n2 = 15000
    w_init = paddle.nn.initializer.XavierUniform()
    Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    assert (
        np.maximum(w, np.ones((n1, n2)) * np.sqrt(6.0 / (n1 + n2))) == np.ones((n1, n2)) * np.sqrt(6.0 / (n1 + n2))
    ).all()
    assert (
        np.minimum(w, np.ones((n1, n2)) * -np.sqrt(6.0 / (n1 + n2))) == np.ones((n1, n2)) * -np.sqrt(6.0 / (n1 + n2))
    ).all()
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(n2), delta=0.3, rtol=0.3)
    compare(np.mean(w, axis=1), np.zeros(n1), delta=0.3, rtol=0.3)


@pytest.mark.api_initializer_xavier_uniform_parameters
def test_initializer_xavier_uniform_1():
    """
    base
    """
    n1 = 16000
    n2 = 15000
    w_init = paddle.nn.initializer.XavierUniform()
    Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    assert (
        np.maximum(w, np.ones((n1, n2)) * np.sqrt(6.0 / (n1 + n2))) == np.ones((n1, n2)) * np.sqrt(6.0 / (n1 + n2))
    ).all()
    assert (
        np.minimum(w, np.ones((n1, n2)) * -np.sqrt(6.0 / (n1 + n2))) == np.ones((n1, n2)) * -np.sqrt(6.0 / (n1 + n2))
    ).all()
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(n2), delta=0.3, rtol=0.3)
    compare(np.mean(w, axis=1), np.zeros(n1), delta=0.3, rtol=0.3)


@pytest.mark.api_initializer_xavier_uniform_parameters
def test_initializer_xavier_uniform_2():
    """
    base
    """
    n1 = 3000
    n2 = 3500
    fan_in = 1000.3
    fan_out = 1200.5
    w_init = paddle.nn.initializer.XavierUniform(fan_in=fan_in, fan_out=fan_out)
    Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    assert (
        np.maximum(w, np.ones((n1, n2)) * np.sqrt(6.0 / (fan_in + fan_out)))
        == np.ones((n1, n2)) * np.sqrt(6.0 / (fan_in + fan_out))
    ).all()
    assert (
        np.minimum(w, np.ones((n1, n2)) * -np.sqrt(6.0 / (fan_in + fan_out)))
        == np.ones((n1, n2)) * -np.sqrt(6.0 / (fan_in + fan_out))
    ).all()
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(n2), delta=0.3, rtol=0.3)
    compare(np.mean(w, axis=1), np.zeros(n1), delta=0.3, rtol=0.3)
