#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_IndependentTransform
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_IndependentTransform_parameters
def test_IndependentTransform0():
    """
    base: ExpTransform
    reinterpreted_batch_rank=1
    """
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    multi_exp = paddle.distribution.IndependentTransform(paddle.distribution.ExpTransform(), 1)

    assert np.allclose(multi_exp.forward(x).numpy(), paddle.exp(x).numpy())
    assert np.allclose(multi_exp.inverse(multi_exp.forward(x)).numpy(), x.numpy())
    assert np.allclose(multi_exp.forward_log_det_jacobian(x).numpy(), np.array([6.0, 15.0]))
    assert np.allclose(multi_exp.inverse_log_det_jacobian(multi_exp.forward(x)).numpy(), np.array([-6.0, -15.0]))


@pytest.mark.api_distribution_Independent_parameters
def test_Independent1():
    """
    base: ExpTransform
    reinterpreted_batch_rank = 2
    """
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    multi_exp = paddle.distribution.IndependentTransform(paddle.distribution.ExpTransform(), 2)

    assert np.allclose(multi_exp.forward(x).numpy(), paddle.exp(x).numpy())
    assert np.allclose(multi_exp.inverse(multi_exp.forward(x)).numpy(), x.numpy())
    assert np.allclose(multi_exp.forward_log_det_jacobian(x).numpy(), np.array([21.0]))
    assert np.allclose(multi_exp.inverse_log_det_jacobian(multi_exp.forward(x)).numpy(), np.array([-21.0]))


@pytest.mark.api_distribution_IndependentTransform_parameters
def test_IndependentTransform2():
    """
    base: PowerTransform
    reinterpreted_batch_rank=1
    """
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mul_power = paddle.distribution.IndependentTransform(paddle.distribution.PowerTransform(paddle.to_tensor([1.0])), 1)

    assert np.allclose(mul_power.forward(x).numpy(), x.numpy())
    assert np.allclose(mul_power.inverse(x).numpy(), x.numpy())
    assert np.allclose(mul_power.forward_log_det_jacobian(x).numpy(), np.array([0.0, 0.0]))
    assert np.allclose(mul_power.inverse_log_det_jacobian(x).numpy(), np.array([0.0, 0.0]))


@pytest.mark.api_distribution_IndependentTransform_parameters
def test_IndependentTransform3():
    """
    base: PowerTransform
    reinterpreted_batch_rank=2
    """
    x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mul_power = paddle.distribution.IndependentTransform(paddle.distribution.PowerTransform(paddle.to_tensor([1.0])), 2)

    assert np.allclose(mul_power.forward(x).numpy(), x.numpy())
    assert np.allclose(mul_power.inverse(x).numpy(), x.numpy())
    assert np.allclose(mul_power.forward_log_det_jacobian(x).numpy(), np.array([0.0]))
    assert np.allclose(mul_power.inverse_log_det_jacobian(x).numpy(), np.array([0.0]))
