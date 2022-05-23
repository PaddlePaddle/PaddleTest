#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fused_feedforward
"""
import sys
from apibase import APIBase
import paddle
import pytest
import numpy as np
from paddle.fluid.framework import _enable_legacy_dygraph

_enable_legacy_dygraph()

sys.path.append("../../utils/")
from interceptor import skip_not_compile_gpu


class TestFusedFeedForward(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True
        self.delta = 1e-2


obj = TestFusedFeedForward(paddle.incubate.nn.FusedFeedForward)
obj.places = [paddle.CUDAPlace(0)]


@skip_not_compile_gpu
@pytest.mark.api_nn_FusedFeedForward_parameters
def test_fused_feedforward0():
    """
    default
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 2)
    res = np.array([[[-0.9997536, 0.9997536], [-0.9999639, 0.9999639]]])
    obj.run(
        res=res,
        data=x,
        d_model=2,
        dim_feedforward=2,
        dropout_rate=0,
        act_dropout_rate=0,
        weight_attr=paddle.nn.initializer.Constant(2),
        bias_attr=paddle.nn.initializer.Constant(2),
    )


@skip_not_compile_gpu
@pytest.mark.api_nn_FusedFeedForward_parameters
def test_fused_feedforward1():
    """
    default
    activation='gelu'
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 3)
    res = np.array([[[-1.3824339, 0.9524618, 0.4299395], [1.3745127, -0.97339916, -0.40110698]]])
    obj.run(
        res=res,
        data=x,
        d_model=3,
        dim_feedforward=3,
        dropout_rate=0,
        act_dropout_rate=0,
        activation="gelu",
        weight_attr=paddle.nn.initializer.Constant(2),
        bias_attr=paddle.nn.initializer.Constant(2),
    )


@skip_not_compile_gpu
@pytest.mark.api_nn_FusedFeedForward_parameters
def test_fused_feedforward2():
    """
    normalize_before=True
    """
    np.random.seed(22)
    x = np.random.rand(1, 2, 2)
    res = np.array([[[10.20846272, 10.48168278], [10.42053699, 10.85918140]]])
    obj.run(
        res=res,
        data=x,
        d_model=2,
        dim_feedforward=2,
        dropout_rate=0,
        act_dropout_rate=0,
        normalize_before=True,
        weight_attr=paddle.nn.initializer.Constant(2),
        bias_attr=paddle.nn.initializer.Constant(2),
    )
