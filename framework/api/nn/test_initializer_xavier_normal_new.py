#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer_xavier_normal
"""
from apibase import APIBase
from apibase import compare
import pytest
import paddle
import numpy as np


@pytest.mark.api_initializer_xavier_normal_vartype
def test_initializer_xavier_normal_base():
    """
    base
    """
    n1 = 10000
    n2 = 10000
    w_init = paddle.nn.initializer.XavierNormal()
    Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    compare(np.array(np.std(w)), 1 * np.sqrt(2 / (n1 + n2)), delta=1e-2, rtol=1e-2)
    compare(np.std(w, axis=0), np.ones(n2) * np.sqrt(2 / (n1 + n2)), delta=1e-1, rtol=1e-1)
    compare(np.std(w, axis=1), np.ones(n1) * np.sqrt(2 / (n1 + n2)), delta=1e-1, rtol=1e-1)
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(n2), delta=0.3, rtol=0.3)
    compare(np.mean(w, axis=1), np.zeros(n1), delta=0.3, rtol=0.3)


@pytest.mark.api_initializer_xavier_normal_parameters
def test_initializer_xavier_normal1():
    """
    base
    """
    n1 = 20000
    n2 = 10000
    w_init = paddle.nn.initializer.XavierNormal()
    Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    compare(np.array(np.std(w)), 1 * np.sqrt(2 / (n1 + n2)), delta=1e-2, rtol=1e-2)
    compare(np.std(w, axis=0), np.ones(n2) * np.sqrt(2 / (n1 + n2)), delta=1e-1, rtol=1e-1)
    compare(np.std(w, axis=1), np.ones(n1) * np.sqrt(2 / (n1 + n2)), delta=1e-1, rtol=1e-1)
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(n2), delta=0.3, rtol=0.3)
    compare(np.mean(w, axis=1), np.zeros(n1), delta=0.3, rtol=0.3)


@pytest.mark.api_initializer_xavier_normal_parameters
def test_initializer_xavier_normal2():
    """
    base
    """
    n1 = 20000
    n2 = 15000
    fan_in = 0.1
    fan_out = 0.2
    w_init = paddle.nn.initializer.XavierNormal(fan_in=fan_in, fan_out=fan_out)
    Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
    w = Linear._parameters["weight"].numpy()

    compare(np.array(np.std(w)), 1 * np.sqrt(2 / (fan_in + fan_out)), delta=1e-2, rtol=1e-2)
    compare(np.std(w, axis=0), np.ones(n2) * np.sqrt(2 / (fan_in + fan_out)), delta=1e-1, rtol=1e-1)
    compare(np.std(w, axis=1), np.ones(n1) * np.sqrt(2 / (fan_in + fan_out)), delta=1e-1, rtol=1e-1)
    compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
    compare(np.mean(w, axis=0), np.zeros(n2), delta=0.3, rtol=0.3)
    compare(np.mean(w, axis=1), np.zeros(n1), delta=0.3, rtol=0.3)


# @pytest.mark.api_initializer_xavier_normal_parameters
# def test_initializer_xavier_normal3():
#     """
#     base
#     """
#     n1 = 20000
#     n2 = 10000
#     fan_in = 0.1
#     fan_out = -0.2
#     w_init = paddle.nn.initializer.XavierNormal(fan_in=fan_in, fan_out=fan_out)
#     Linear = paddle.nn.Linear(n1, n2, weight_attr=w_init)
#     w = Linear._parameters['weight'].numpy()
#
#     compare(np.array(np.std(w)), 1 * np.sqrt(2 / (fan_in + fan_out)), delta=1e-2, rtol=1e-2)
#     compare(np.std(w, axis=0), np.ones(n2) * np.sqrt(2 / (fan_in + fan_out)), delta=1e-1, rtol=1e-1)
#     compare(np.std(w, axis=1), np.ones(n1) * np.sqrt(2 / (fan_in + fan_out)), delta=1e-1, rtol=1e-1)
#     compare(np.array(np.mean(w)), 0, delta=1e-2, rtol=1e-2)
#     compare(np.mean(w, axis=0), np.zeros(n2), delta=1e-1, rtol=1e-1)
#     compare(np.mean(w, axis=1), np.zeros(n1), delta=1e-1, rtol=1e-1)
