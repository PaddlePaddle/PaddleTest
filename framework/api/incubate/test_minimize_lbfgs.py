#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test minimize_lbfgs
"""

import paddle
import pytest
import numpy as np


def func0(x):
    """
    func0
    """
    return paddle.dot(x, x)


def func1(x):
    """
    func1
    """
    return paddle.mean(paddle.nn.functional.sigmoid(x))


@pytest.mark.api_incubate_minimize_lbfgs_vartype
def test_minimize_lbfgs0():
    """
    default
    """
    x0 = paddle.to_tensor([1.1, 2.2, 1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func0, x0)
    res = paddle.load("./data/lbfgs_data/func0_default")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_lbfgs_parameters
def test_minimize_lbfgs1():
    """
    func1
    """
    x0 = paddle.to_tensor([1.1, 2.2, 1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func1, x0)
    res = paddle.load("./data/lbfgs_data/func1_default")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_lbfgs_parameters
def test_minimize_lbfgs2():
    """
    max_iters=1
    """
    x0 = paddle.to_tensor([1.1, 2.2, 1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func1, x0, max_iters=1)
    res = paddle.load("./data/lbfgs_data/func1_iters1")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_lbfgs_parameters
def test_minimize_lbfgs3():
    """
    max_line_search_iters=2
    """
    x0 = paddle.to_tensor([1.1, 2.2, 1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func1, x0, max_line_search_iters=2)
    res = paddle.load("./data/lbfgs_data/func1_lineiters2")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_lbfgs_parameters
def test_minimize_lbfgs4():
    """
    initial_step_length=0.4
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2])
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func1, x0, initial_step_length=0.4)
    res = paddle.load("./data/lbfgs_data/func1_step0.4")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_lbfgs_parameters
def test_minimize_lbfgs5():
    """
    dtype=float64
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2], dtype="float64")
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func1, x0, dtype="float64")
    res = paddle.load("./data/lbfgs_data/func1_fp64")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_lbfgs_parameters
def test_minimize_lbfgs6():
    """
    func1
    kwargs
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2], dtype="float64")
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(
        func1, x0, max_iters=1, max_line_search_iters=2, initial_step_length=1.2, dtype="float64"
    )
    res = paddle.load("./data/lbfgs_data/func1_kwargs")
    for i in range(5):
        assert np.allclose(res[i].numpy(), results[i].numpy())
