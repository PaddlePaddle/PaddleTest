#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_AbsTransform
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_AbsTransform_parameters
def test_AbsTransform0():
    """
    test
    """
    abs = paddle.distribution.AbsTransform()
    forward_value = abs.forward(paddle.to_tensor([-1, 0, 1]))
    inverse_value = abs.inverse(paddle.to_tensor(1))
    inverse_ldj = abs.inverse_log_det_jacobian(paddle.to_tensor([1.0, 2.0, 3.0]))
    assert np.allclose(forward_value.numpy(), np.array([1, 0, 1]))
    assert np.allclose(inverse_value[0] + inverse_value[1], np.zeros(1))
    assert np.allclose(inverse_ldj[0], np.array([0]))
    assert np.allclose(inverse_ldj[1], np.array([0]))
