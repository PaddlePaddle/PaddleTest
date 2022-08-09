#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_PowerTransform
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_PowerTransform_parameters
def test_PowerTransform0():
    """
    power = 1.
    """
    x = paddle.to_tensor([1.0, 2.0])
    power = paddle.distribution.PowerTransform(paddle.to_tensor(1.0))

    assert np.allclose(power.forward(x).numpy(), x)
    assert np.allclose(power.inverse(power.forward(x)).numpy(), x)
    assert np.allclose(power.forward_log_det_jacobian(x), np.array([0.0, 0.0]))
    assert np.allclose(power.inverse_log_det_jacobian(x), np.array([0.0, 0.0]))


@pytest.mark.api_distribution_PowerTransform_parameters
def test_PowerTransform1():
    """
    power = 0.
    x: 2-d tensor
    """
    x = paddle.to_tensor([[1.0, 2.0]])
    power = paddle.distribution.PowerTransform(paddle.to_tensor(0.0))

    assert np.allclose(power.forward(x).numpy(), np.array([[1.0, 1.0]]))
    assert np.allclose(power.inverse(power.forward(x)).numpy(), np.array([[1.0, 1.0]]))
    assert np.allclose(power.forward_log_det_jacobian(x), np.array([[-np.inf, -np.inf]]))
    assert np.allclose(power.inverse_log_det_jacobian(x), np.array([[np.inf, np.inf]]))


@pytest.mark.api_distribution_PowerTransform_parameters
def test_PowerTransform2():
    """
    power = -1.
    x: 3-d tensor
    """
    x = paddle.to_tensor([[1.0, 2.0]])
    power = paddle.distribution.PowerTransform(paddle.to_tensor(-1.0))

    assert np.allclose(power.forward(x).numpy(), np.array([[[1.0, 0.50000000]]]))
    assert np.allclose(power.inverse(power.forward(x)).numpy(), x)
    assert np.allclose(power.forward_log_det_jacobian(x), np.array([[[0.0, -1.38629436]]]))
    assert np.allclose(power.inverse_log_det_jacobian(x), np.array([[[0.0, -1.38629436]]]))
