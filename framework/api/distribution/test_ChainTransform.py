#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ChainTransform
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_ChainTransform_parameters
def test_ChainTransform0():
    """
    AffineTransform and ExpTransform
    """
    x = paddle.to_tensor([0.0, 1.0, 2.0, 3.0])
    chain = paddle.distribution.ChainTransform(
        (
            paddle.distribution.AffineTransform(paddle.to_tensor(0.0), paddle.to_tensor(1.0)),
            paddle.distribution.ExpTransform(),
        )
    )

    forward_value = chain.forward(x)
    inverse_value = chain.inverse(chain.forward(x))
    forward_ldj = chain.forward_log_det_jacobian(x)
    inverse_ldj = chain.inverse_log_det_jacobian(chain.forward(x))
    assert np.allclose(forward_value.numpy(), np.exp(x.numpy()))
    assert np.allclose(inverse_value.numpy(), x.numpy())
    assert np.allclose(forward_ldj, x.numpy())
    assert np.allclose(inverse_ldj, -x.numpy())


@pytest.mark.api_distribution_ChainTransform_parameters
def test_ChainTransform1():
    """
    AffineTransform 、 ExpTransform 、 AbsTransform
    """
    x = paddle.to_tensor([0.0, 1.0, 2.0, 3.0])
    chain = paddle.distribution.ChainTransform(
        (
            paddle.distribution.AffineTransform(paddle.to_tensor(0.0), paddle.to_tensor(1.0)),
            paddle.distribution.ExpTransform(),
            paddle.distribution.PowerTransform(paddle.to_tensor(1.0)),
        )
    )

    forward_value = chain.forward(x)
    inverse_value = chain.inverse(chain.forward(x))
    forward_ldj = chain.forward_log_det_jacobian(x)
    inverse_ldj = chain.inverse_log_det_jacobian(chain.forward(x))
    assert np.allclose(forward_value.numpy(), np.exp(x.numpy()))
    assert np.allclose(inverse_value.numpy(), x.numpy())
    assert np.allclose(forward_ldj, x.numpy())
    assert np.allclose(inverse_ldj, -x.numpy())
