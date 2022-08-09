#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_TransformedDistribution
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_TransformedDistribution_parameters
def test_TransformedDistribution0():
    """
    base: Normal
    transforms: AffineTransform
    """
    d = paddle.distribution.TransformedDistribution(
        paddle.distribution.Normal(0.0, 1.0),
        [paddle.distribution.AffineTransform(paddle.to_tensor(1.0), paddle.to_tensor(2.0))],
    )

    prob_value = d.prob(paddle.to_tensor([2.0, 1.0]))
    log_prob_value = d.log_prob(paddle.to_tensor(0.5))
    assert d.batch_shape == (1,)
    assert d.event_shape == (1,)
    assert np.allclose(prob_value.numpy(), np.array([0.07022688]))
    assert np.allclose(log_prob_value.numpy(), np.array([-1.64333570]))


@pytest.mark.api_distribution_TransformedDistribution_parameters
def test_TransformedDistribution1():
    """
    base: Multinomial
    transforms: AffineTransform
    """
    d = paddle.distribution.TransformedDistribution(
        paddle.distribution.Multinomial(10, paddle.to_tensor([0.2, 0.3, 0.5])),
        [paddle.distribution.AffineTransform(paddle.to_tensor(1.0), paddle.to_tensor(2.0))],
    )

    prob_value = d.prob(paddle.to_tensor([0.2]))
    log_prob_value = d.log_prob(paddle.to_tensor(1))
    assert not d.batch_shape
    assert not d.event_shape
    assert np.allclose(prob_value.numpy(), np.array([3.58325624]))
    assert np.allclose(log_prob_value.numpy(), np.array([-0.69314718]))


@pytest.mark.api_distribution_TransformedDistribution_parameters
def test_TransformedDistribution2():
    """
    base: Normal
    transforms: AffineTransform,ExpTransform
    """
    d = paddle.distribution.TransformedDistribution(
        paddle.distribution.Normal(0.0, 1.0),
        [
            paddle.distribution.AffineTransform(paddle.to_tensor(1.0), paddle.to_tensor(2.0)),
            paddle.distribution.ExpTransform(),
        ],
    )

    prob_value = d.prob(paddle.to_tensor([0.2]))
    log_prob_value = d.log_prob(paddle.to_tensor([0.2, 0.3]))
    assert d.batch_shape == (1,)
    assert d.event_shape == (1,)
    assert np.allclose(prob_value.numpy(), np.array([0.42579654]))
    assert np.allclose(log_prob_value.numpy(), np.array([-1.17594624]))


@pytest.mark.api_distribution_TransformedDistribution_parameters
def test_TransformedDistribution3():
    """
    base: Normal
    transforms: AffineTransform,SigmoidTransform
    """
    d = paddle.distribution.TransformedDistribution(
        paddle.distribution.Normal(0.0, 1.0),
        [
            paddle.distribution.AffineTransform(paddle.to_tensor(1.0), paddle.to_tensor(2.0)),
            paddle.distribution.SigmoidTransform(),
        ],
    )

    prob_value = d.prob(paddle.to_tensor([0.2]))
    log_prob_value = d.log_prob(paddle.to_tensor([0.2, 0.3]))
    assert d.batch_shape == (1,)
    assert d.event_shape == (1,)
    assert np.allclose(prob_value.numpy(), np.array([0.61182785]))
    assert np.allclose(log_prob_value.numpy(), np.array([-0.27615857]))


@pytest.mark.api_distribution_TransformedDistribution_parameters
def test_TransformedDistribution4():
    """
    base: Normal
    transforms: AffineTransform,SigmoidTransform,TanhTransform
    """
    d = paddle.distribution.TransformedDistribution(
        paddle.distribution.Normal(0.0, 1.0),
        [
            paddle.distribution.AffineTransform(paddle.to_tensor(1.0), paddle.to_tensor(2.0)),
            paddle.distribution.SigmoidTransform(),
            paddle.distribution.TanhTransform(),
        ],
    )

    prob_value = d.prob(paddle.to_tensor([0.2]))
    log_prob_value = d.log_prob(paddle.to_tensor([0.2, 0.3]))
    assert d.batch_shape == (1,)
    assert d.event_shape == (1,)
    assert np.allclose(prob_value.numpy(), np.array([0.63729006]))
    assert np.allclose(log_prob_value.numpy(), np.array([-0.13812208]))
