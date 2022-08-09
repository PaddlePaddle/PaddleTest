#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Independent
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_Independent_parameters
def test_Independent0():
    """
    base: Beta
    """
    beta = paddle.distribution.Beta(paddle.to_tensor([0.5, 0.5]), paddle.to_tensor([0.5, 0.5]))
    reinterpreted_beta = paddle.distribution.Independent(beta, 1)

    prob_value = reinterpreted_beta.prob(paddle.to_tensor([0.9]))
    log_prob_value = reinterpreted_beta.log_prob(paddle.to_tensor([0.2]))
    etpy = reinterpreted_beta.entropy()

    assert beta.batch_shape == reinterpreted_beta.event_shape
    assert beta.event_shape == reinterpreted_beta.batch_shape
    assert np.allclose(prob_value.numpy(), np.array([1.12579060]))
    assert np.allclose(log_prob_value.numpy(), np.array([-0.45687842]))
    assert np.allclose(etpy.numpy(), np.array([-0.48312974]))


@pytest.mark.api_distribution_Independent_parameters
def test_Independent1():
    """
    base: Beta
    reinterpreted_batch_rank = 2
    """
    beta = paddle.distribution.Beta(
        paddle.to_tensor([[0.5, 0.5], [0.2, 0.2]]), paddle.to_tensor([[0.5, 0.5], [0.1, 0.1]])
    )
    reinterpreted_beta = paddle.distribution.Independent(beta, 2)

    prob_value = reinterpreted_beta.prob(paddle.to_tensor([0.9]))
    log_prob_value = reinterpreted_beta.log_prob(paddle.to_tensor([0.2]))
    etpy = reinterpreted_beta.entropy()

    assert beta.batch_shape == reinterpreted_beta.event_shape
    assert beta.event_shape == reinterpreted_beta.batch_shape
    assert np.allclose(prob_value.numpy(), np.array([0.39445820]))
    assert np.allclose(log_prob_value.numpy(), np.array([-2.84207654]))
    assert np.allclose(etpy.numpy(), np.array([-10.43781281]))


@pytest.mark.api_distribution_Independent_parameters
def test_Independent2():
    """
    base: Multinomial
    """
    mul = paddle.distribution.Multinomial(10, paddle.to_tensor([[0.2, 0.3], [0.5, 0.6]]))
    reinterpreted_mul = paddle.distribution.Independent(mul, 1)

    prob_value = reinterpreted_mul.prob(paddle.to_tensor([0.9]))
    log_prob_value = reinterpreted_mul.log_prob(paddle.to_tensor([0.2]))
    etpy = reinterpreted_mul.entropy()

    assert reinterpreted_mul.batch_shape == ()
    assert reinterpreted_mul.event_shape == (2, 2)
    assert np.allclose(prob_value.numpy(), np.array([0.25918952]))
    assert np.allclose(log_prob_value.numpy(), np.array([-0.46207130]))
    assert np.allclose(etpy.numpy(), np.array([3.72496510]))


@pytest.mark.api_distribution_Independent_parameters
def test_Independent3():
    """
    base: Multinomial
    reinterpreted_batch_rank = 2
    """
    mul = paddle.distribution.Multinomial(10, paddle.to_tensor([[[0.2, 0.3]], [[0.5, 0.6]]]))
    reinterpreted_mul = paddle.distribution.Independent(mul, 2)

    prob_value = reinterpreted_mul.prob(paddle.to_tensor([0.9]))
    log_prob_value = reinterpreted_mul.log_prob(paddle.to_tensor([0.2]))
    etpy = reinterpreted_mul.entropy()

    assert reinterpreted_mul.batch_shape == ()
    assert reinterpreted_mul.event_shape == (2, 1, 2)
    assert np.allclose(prob_value.numpy(), np.array([0.25918952]))
    assert np.allclose(log_prob_value.numpy(), np.array([-0.46207130]))
    assert np.allclose(etpy.numpy(), np.array([3.72496510]))
