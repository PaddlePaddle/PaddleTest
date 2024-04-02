#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Multinomial
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_multinomial_parameters
def test_multinomial0():
    """
    test01
    """
    x = np.array([0.2, 0.3, 0.5])
    probs = paddle.to_tensor(x)
    multinomial = paddle.distribution.Multinomial(10, probs)
    res = multinomial.sample((2,)).numpy()
    assert res.shape == (2, 3)
    for i in range(2):
        assert np.sum(res[i]) == 10
    assert np.allclose(multinomial.mean.numpy(), np.array([2, 3, 5]))
    assert np.allclose(multinomial.variance.numpy(), np.array([1.60000002, 2.09999990, 2.50000000]))
    assert np.allclose(multinomial.prob(paddle.to_tensor([0, 1, 2])), np.array([0.225]))
    assert np.allclose(multinomial.entropy().numpy(), np.array(3.34121895))


@pytest.mark.api_distribution_multinomial_parameters
def test_multinomial1():
    """
    test02
    """
    x = np.array([0.4, 0.1, 0.3, 0.2])
    probs = paddle.to_tensor(x)
    multinomial = paddle.distribution.Multinomial(14, probs)
    res = multinomial.sample((2, 4)).numpy()
    assert res.shape == (2, 4, 4)
    for i in range(2):
        for j in range(4):
            assert np.sum(res[i][j]) == 14


@pytest.mark.api_distribution_multinomial_parameters
def test_multinomial2():
    """
    test03
    probs sum > 1
    """
    x = np.array([0.4, 0.1, 0.7])
    probs = paddle.to_tensor(x)
    multinomial = paddle.distribution.Multinomial(14, probs)
    res = multinomial.sample((2, 4)).numpy()
    assert res.shape == (2, 4, 3)
    for i in range(2):
        for j in range(4):
            assert np.sum(res[i][j]) == 14


@pytest.mark.api_distribution_multinomial_parameters
def test_multinomial3():
    """
    test03
    probs 2d-tensor
    """
    x = np.array([[0.4, 0.1, 0.5], [0.3, 0.2, 0.5]])
    probs = paddle.to_tensor(x)
    multinomial = paddle.distribution.Multinomial(14, probs)
    res = multinomial.sample((2, 4)).numpy()
    assert res.shape == (2, 4, 2, 3)
    for i in range(2):
        for j in range(4):
            assert np.sum(res[i][j]) == 28
