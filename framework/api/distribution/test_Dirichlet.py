#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Dirichlet
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_dirichlet_parameters
def test_dirichlet0():
    """
    test01
    """
    concentration = paddle.to_tensor([1.0, 2.0, 3.0])
    dirichlet = paddle.distribution.Dirichlet(concentration)
    assert np.allclose(dirichlet.mean, np.array([0.16666667, 0.33333334, 0.50000000]))
    assert np.allclose(dirichlet.variance, np.array([0.01984127, 0.03174603, 0.03571429]))
    val = paddle.to_tensor([0.3, 0.5, 0.6])
    assert np.allclose(dirichlet.prob(val), np.array([10.80000019]))
    assert np.allclose(dirichlet.log_prob(val), np.array([2.37954617]))
    assert np.allclose(dirichlet.entropy(), np.array([-1.24434423]))


@pytest.mark.api_distribution_dirichlet_parameters
def test_dirichlet1():
    """
    test02
    """
    concentration = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dirichlet = paddle.distribution.Dirichlet(concentration)
    assert np.allclose(
        dirichlet.mean, np.array([[0.16666667, 0.33333334, 0.50000000], [0.26666668, 0.33333334, 0.40000001]])
    )
    assert np.allclose(
        dirichlet.variance, np.array([[0.01984127, 0.03174603, 0.03571429], [0.01222222, 0.01388889, 0.01500000]])
    )
    val = paddle.to_tensor([0.3, 0.5, 0.6])
    assert np.allclose(dirichlet.prob(val), np.array([10.80000019, 662.01037598]))
    assert np.allclose(dirichlet.log_prob(val), np.array([2.37954617, 6.49528122]))
    assert np.allclose(dirichlet.entropy(), np.array([-1.24434423, -1.66516876]))
