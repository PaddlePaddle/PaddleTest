#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_kl_divergence
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_kl_divergence_parameters
def test_kl_divergence0():
    """
    test01
    """
    beta1 = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    beta2 = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    kl_divergence = paddle.distribution.kl_divergence(beta1, beta2)
    assert np.allclose(kl_divergence, np.array([0]))


@pytest.mark.api_distribution_kl_divergence_parameters
def test_kl_divergence1():
    """
    test02
    """
    beta1 = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    beta2 = paddle.distribution.Beta(alpha=0.2, beta=1)
    kl_divergence = paddle.distribution.kl_divergence(beta1, beta2)
    assert np.allclose(kl_divergence, np.array([0.74196696]))


@pytest.mark.api_distribution_kl_divergence_parameters
def test_kl_divergence2():
    """
    test03
    """
    beta1 = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    beta2 = paddle.distribution.Beta(alpha=paddle.to_tensor([1.0, 2.0]), beta=0.7)
    kl_divergence = paddle.distribution.kl_divergence(beta1, beta2)
    assert np.allclose(kl_divergence, np.array([0.18235138, 1.03801811]))


@pytest.mark.api_distribution_kl_divergence_parameters
def test_kl_divergence3():
    """
    test04
    """
    dirichlet1 = paddle.distribution.Dirichlet(paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    dirichlet2 = paddle.distribution.Dirichlet(paddle.to_tensor([0.5, 0.2, 0.4]))
    kl_divergence = paddle.distribution.kl_divergence(dirichlet1, dirichlet2)
    assert np.allclose(kl_divergence, np.array([1.54899025, 2.38351250]))
