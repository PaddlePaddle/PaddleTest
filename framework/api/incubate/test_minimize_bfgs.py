#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test minimize_bfgs
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
    return paddle.sum(paddle.tanh(x))


def func2(x):
    """
    func2
    """
    return paddle.mean(paddle.nn.functional.sigmoid(x))


@pytest.mark.api_incubate_minimize_bfgs_vartype
def test_minimize_bfgs0():
    """
    default
    """
    x0 = paddle.to_tensor([1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func0, x0)
    res = paddle.load("./data/bfgs_data/func0_default")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs1():
    """
    max_iters=1
    """
    x0 = paddle.to_tensor([1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func0, x0, max_iters=1)
    res = paddle.load("./data/bfgs_data/func0_iter1")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs2():
    """
    max_line_search_iters=1
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func0, x0, max_line_search_iters=1)
    res = paddle.load("./data/bfgs_data/func0_liniters1")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs3():
    """
    initial_step_length=0.4
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func0, x0, initial_step_length=0.4)
    res = paddle.load("./data/bfgs_data/func0_step0.4")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs4():
    """
    dtype=float64
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2], dtype="float64")
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func0, x0, dtype="float64")
    res = paddle.load("./data/bfgs_data/func0_fp64")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs5():
    """
    func1
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func1, x0)
    res = paddle.load("./data/bfgs_data/func1_default")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs6():
    """
    func1
    set kwargs
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2], dtype="float64")
    results = paddle.incubate.optimizer.functional.minimize_bfgs(
        func1, x0, max_iters=1, max_line_search_iters=4, initial_step_length=2.0, dtype="float64"
    )
    res = paddle.load("./data/bfgs_data/func1_kwargs")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs7():
    """
    func2
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func2, x0)
    res = paddle.load("./data/bfgs_data/func2_default")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())


@pytest.mark.api_incubate_minimize_bfgs_parameters
def test_minimize_bfgs8():
    """
    func2
    set kwargs
    """
    x0 = paddle.to_tensor([2.3, 4.4, 5.1, 1.2], dtype="float64")
    results = paddle.incubate.optimizer.functional.minimize_bfgs(
        func2, x0, max_iters=1, max_line_search_iters=2, initial_step_length=1.2, dtype="float64"
    )
    res = paddle.load("./data/bfgs_data/func2_kwargs")
    for i in range(6):
        assert np.allclose(res[i].numpy(), results[i].numpy())
