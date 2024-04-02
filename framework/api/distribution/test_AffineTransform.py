#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_AffineTransform
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_AffineTransform_parameters
def test_AffineTransform0():
    """
    loc = 0, scale=1
    """
    x = paddle.to_tensor([1.0, 2.0])
    affine = paddle.distribution.AffineTransform(paddle.to_tensor(0.0), paddle.to_tensor(1.0))

    assert np.allclose(affine.forward(x).numpy(), np.array([1, 2]))
    assert np.allclose(affine.inverse(x).numpy(), np.array([1, 2]))
    assert np.allclose(affine.forward_log_det_jacobian(x), np.array([0]))
    assert np.allclose(affine.inverse_log_det_jacobian(x), np.array([0]))


@pytest.mark.api_distribution_AffineTransform_parameters
def test_AffineTransform1():
    """
    loc = 0.5, scale=0.5
    x: 2d-tensor
    """
    x = paddle.rand((3, 4))
    affine = paddle.distribution.AffineTransform(paddle.to_tensor(0.5), paddle.to_tensor(0.5))

    assert np.allclose(affine.forward(x).numpy(), 0.5 * x + 0.5)
    assert np.allclose(affine.inverse(x).numpy(), (x - 0.5) / 0.5)
    assert np.allclose(affine.forward_log_det_jacobian(x).numpy(), np.array([-0.69314718]))
    assert np.allclose(affine.inverse_log_det_jacobian(x), np.array([0.69314718]))


@pytest.mark.api_distribution_AffineTransform_parameters
def test_AffineTransform2():
    """
    loc = 1., scale=0.
    x: 3d-tensor
    """
    x = paddle.rand((3, 4, 5))
    affine = paddle.distribution.AffineTransform(paddle.to_tensor(1.0), paddle.to_tensor(0.0))

    assert np.allclose(affine.forward(x).numpy(), np.ones_like(x))
    assert np.allclose(affine.inverse(x).numpy(), np.ones_like(x) - np.inf)
    assert np.allclose(affine.forward_log_det_jacobian(x).numpy(), np.array([-np.inf]))
    assert np.allclose(affine.inverse_log_det_jacobian(x), np.array([np.inf]))
