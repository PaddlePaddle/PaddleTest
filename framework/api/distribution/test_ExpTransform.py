#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ExpTransform
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_ExpTransform_parameters
def test_ExpTransform0():
    """
    test
    """
    exp = paddle.distribution.ExpTransform()
    x = paddle.to_tensor([1.0, 0.0, 1.0])
    forward_value = exp.forward(x)
    inverse_value = exp.inverse(exp.forward(x))
    forward_ldj = exp.forward_log_det_jacobian(x)
    inverse_ldj = exp.inverse_log_det_jacobian(x)
    assert np.allclose(forward_value.numpy(), np.array([2.71828175, 1.0, 2.71828175]))
    assert np.allclose(inverse_value, x)
    assert np.allclose(forward_ldj, x)
    assert np.allclose(inverse_ldj, np.array([0.0, np.inf, 0.0]))
