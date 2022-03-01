#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Beta
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_beta_parameters
def test_beta0():
    """
    test01
    """
    beta = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    assert np.allclose(beta.mean, np.array([0.5]))
    assert np.allclose(beta.variance, np.array([0.125]))
    val = paddle.to_tensor([0.8])
    assert np.allclose(beta.prob(val), np.array([0.79577476]))
    assert np.allclose(beta.log_prob(val), np.array([-0.22843909]))
    assert np.allclose(beta.entropy(), np.array([-0.24156499]))


@pytest.mark.api_distribution_beta_parameters
def test_beta1():
    """
    test02
    """
    alpha = paddle.to_tensor([0.4, 2])
    beta = paddle.distribution.Beta(alpha=alpha, beta=0.5)
    assert np.allclose(beta.mean, np.array([0.44444448, 0.8]))
    assert np.allclose(beta.variance, np.array([0.12995453, 0.04571429]))
    val = paddle.to_tensor([0.8])
    assert np.allclose(beta.prob(val), np.array([0.69484848, 1.34164083]))
    assert np.allclose(beta.log_prob(val), np.array([-0.36406147, 0.29389340]))
    assert np.allclose(beta.entropy(), np.array([-0.38550019, -0.76527911]))


@pytest.mark.api_distribution_beta_parameters
def test_beta2():
    """
    test03
    """
    alpha = paddle.to_tensor([0.4, 2])
    p_beta = paddle.to_tensor([4, 0.5])
    beta = paddle.distribution.Beta(alpha=alpha, beta=p_beta)
    assert np.allclose(beta.mean, np.array([0.09090909, 0.80000001]))
    assert np.allclose(beta.variance, np.array([0.01530456, 0.04571429]))
    val = paddle.to_tensor([0.8])
    assert np.allclose(beta.prob(val), np.array([0.00696567, 1.34164083]))
    assert np.allclose(beta.log_prob(val), np.array([-4.96676111, 0.29389340]))
    assert np.allclose(beta.entropy(), np.array([-1.76000524, -0.76527911]))


@pytest.mark.api_distribution_beta_parameters
def test_beta3():
    """
    test04
    """
    alpha = paddle.to_tensor([0.4, 2])
    p_beta = paddle.to_tensor([[4, 0.5], [4, 7]])
    beta = paddle.distribution.Beta(alpha=alpha, beta=p_beta)
    assert np.allclose(beta.mean, np.array([[0.09090909, 0.80000001], [0.09090909, 0.22222222]]))
    assert np.allclose(beta.variance, np.array([[0.01530456, 0.04571429], [0.01530456, 0.01728395]]))
    val = paddle.to_tensor([0.8])
    assert np.allclose(beta.prob(val), np.array([[0.00696567, 1.34164083], [0.00696567, 0.00286720]]))
    assert np.allclose(beta.log_prob(val), np.array([[-4.96676111, 0.29389340], [-4.96676111, -5.85441971]]))
    assert np.allclose(beta.entropy(), np.array([[-1.76000524, -0.76527911], [-1.76000524, -0.70035076]]))


@pytest.mark.api_distribution_beta_parameters
def test_beta4():
    """
    test05
    """
    alpha = paddle.to_tensor([0.4, 2])
    p_beta = paddle.to_tensor([[4, 0.5], [4, 7]])
    beta = paddle.distribution.Beta(alpha=alpha, beta=p_beta)
    assert np.allclose(beta.mean, np.array([[0.09090909, 0.80000001], [0.09090909, 0.22222222]]))
    assert np.allclose(beta.variance, np.array([[0.01530456, 0.04571429], [0.01530456, 0.01728395]]))
    val = paddle.to_tensor([0.8, 0.5])
    assert np.allclose(beta.prob(val), np.array([[0.00696567, 0.53033012], [0.00696567, 0.43749991]]))
    assert np.allclose(beta.log_prob(val), np.array([[-4.96676111, -0.63425565], [-4.96676111, -0.82667875]]))
    assert np.allclose(beta.entropy(), np.array([[-1.76000524, -0.76527911], [-1.76000524, -0.70035076]]))
