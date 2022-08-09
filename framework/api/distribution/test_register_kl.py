#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_register_kl
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_distribution_register_kl_parameters
def test_register_kl0():
    """
    test01
    """

    @paddle.distribution.register_kl(paddle.distribution.Beta, paddle.distribution.Beta)
    def kl_beta_beta(p, q):
        return p.mean + q.mean

    beta1 = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    beta2 = paddle.distribution.Beta(alpha=0.5, beta=0.5)
    kl_divergence = paddle.distribution.kl_divergence(beta1, beta2)
    assert np.allclose(kl_divergence, np.array([1]))
