#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fused_feedforward
"""
import sys
import copy
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_not_compile_gpu


class TestFusedFeedforward(APIBase):
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
        self.delta = 1e-3


obj = TestFusedFeedforward(paddle.incubate.nn.functional.fused_feedforward)
obj.places = [paddle.CUDAPlace(0)]


def cal_fused_feedforward(
    x,
    linear1_weight,
    linear2_weight,
    linear1_bias=None,
    linear2_bias=None,
    ln1_scale=None,
    ln1_bias=None,
    ln2_scale=None,
    ln2_bias=None,
    dropout1_rate=0.5,
    dropout2_rate=0.5,
    activation="relu",
    ln1_epsilon=1e-5,
    ln2_epsilon=1e-5,
    pre_layer_norm=False,
    mode="upscale_in_train",
):
    """
    calculate fused_feedforward
    """
    paddle.set_device("gpu:0")
    paddle.disable_static()
    x = paddle.to_tensor(x, dtype="float32")
    linear1_weight, linear2_weight = (
        paddle.to_tensor(linear1_weight, dtype="float32"),
        paddle.to_tensor(linear2_weight, dtype="float32"),
    )
    parameters = [linear1_bias, linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias]
    for i in range(6):
        if parameters[i] is not None:
            parameters[i] = paddle.to_tensor(parameters[i], dtype="float32")
    [linear1_bias, linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias] = parameters
    residual = copy.deepcopy(x)
    src = x
    if pre_layer_norm:

        src = paddle.nn.functional.layer_norm(src, src.shape[-1], epsilon=ln1_epsilon, weight=ln1_scale, bias=ln1_bias)

    src = paddle.nn.functional.linear(src, linear1_weight, bias=linear1_bias)
    if activation == "relu":
        src = paddle.nn.functional.relu(src)
    elif activation == "gelu":
        src = paddle.nn.functional.gelu(src)
    drop1 = paddle.nn.Dropout(p=dropout1_rate, mode=mode)
    src = paddle.nn.functional.linear(drop1(src), linear2_weight, bias=linear2_bias)
    drop2 = paddle.nn.Dropout(p=dropout2_rate, mode=mode)
    src = residual + drop2(src)

    if not pre_layer_norm:
        src = paddle.nn.functional.layer_norm(src, src.shape[-1], epsilon=ln2_epsilon, weight=ln2_scale, bias=ln2_bias)
    return src.numpy()


@skip_not_compile_gpu
@pytest.mark.api_nn_fused_feedforward_parameters
def test_fused_feedforward0():
    """
    default
    """
    x = np.random.rand(1, 2, 2)
    w1 = np.random.rand(2, 2)
    w2 = np.random.rand(2, 2)
    res = cal_fused_feedforward(x, w1, w2, dropout1_rate=0, dropout2_rate=0)
    obj.run(res=res, x=x, linear1_weight=w1, linear2_weight=w2, dropout1_rate=0, dropout2_rate=0)


@skip_not_compile_gpu
@pytest.mark.api_nn_fused_feedforward_parameters
def test_fused_feedforward1():
    """
    pre_layer_norm = True
    """
    x = np.random.rand(1, 2, 2)
    w1 = np.random.rand(2, 2)
    w2 = np.random.rand(2, 2)
    res = cal_fused_feedforward(x, w1, w2, dropout1_rate=0, dropout2_rate=0, pre_layer_norm=True)
    obj.run(res=res, x=x, linear1_weight=w1, linear2_weight=w2, dropout1_rate=0, dropout2_rate=0, pre_layer_norm=True)


@skip_not_compile_gpu
@pytest.mark.api_nn_fused_feedforward_parameters
def test_fused_feedforward2():
    """
    set linear1_bias and linear2_bias
    """
    x = np.random.rand(1, 2, 2)
    w1 = np.random.rand(2, 4)
    b1 = np.random.rand(4)
    w2 = np.random.rand(4, 2)
    b2 = np.random.rand(2)
    res = cal_fused_feedforward(x, w1, w2, linear1_bias=b1, linear2_bias=b2, dropout1_rate=0, dropout2_rate=0)
    obj.run(
        res=res,
        x=x,
        linear1_weight=w1,
        linear2_weight=w2,
        linear1_bias=b1,
        linear2_bias=b2,
        dropout1_rate=0,
        dropout2_rate=0,
    )


@skip_not_compile_gpu
@pytest.mark.api_nn_fused_feedforward_parameters
def test_fused_feedforward3():
    """
    set linear1_bias and linear2_bias
    set ln2_scale and ln2_bias
    """
    obj.enable_backward = False
    x = np.random.rand(1, 2, 2)
    w1 = np.random.rand(2, 4)
    b1 = np.random.rand(4)
    w2 = np.random.rand(4, 2)
    b2 = np.random.rand(2)
    l2_s = np.random.rand(2)
    l2_b = np.random.rand(2)
    res = cal_fused_feedforward(
        x, w1, w2, linear1_bias=b1, linear2_bias=b2, ln2_scale=l2_s, ln2_bias=l2_b, dropout1_rate=0, dropout2_rate=0
    )
    obj.run(
        res=res,
        x=x,
        linear1_weight=w1,
        linear2_weight=w2,
        linear1_bias=b1,
        linear2_bias=b2,
        ln2_scale=l2_s,
        ln2_bias=l2_b,
        dropout1_rate=0,
        dropout2_rate=0,
    )


@skip_not_compile_gpu
@pytest.mark.api_nn_fused_feedforward_parameters
def test_fused_feedforward4():
    """
    set linear1_bias and linear2_bias
    pre_layer_norm = True
    set ln1_scale and ln1_bias
    """
    obj.enable_backward = True
    obj.delta = 1e-2
    x = np.random.rand(1, 2, 2)
    w1 = np.random.rand(2, 4)
    b1 = np.random.rand(4)
    w2 = np.random.rand(4, 2)
    b2 = np.random.rand(2)
    l1_s = np.random.rand(2)
    l1_b = np.random.rand(2)
    res = cal_fused_feedforward(
        x,
        w1,
        w2,
        linear1_bias=b1,
        linear2_bias=b2,
        ln1_scale=l1_s,
        ln1_bias=l1_b,
        dropout1_rate=0,
        dropout2_rate=0,
        pre_layer_norm=True,
    )
    obj.run(
        res=res,
        x=x,
        linear1_weight=w1,
        linear2_weight=w2,
        linear1_bias=b1,
        linear2_bias=b2,
        ln1_scale=l1_s,
        ln1_bias=l1_b,
        dropout1_rate=0,
        dropout2_rate=0,
        pre_layer_norm=True,
    )


@skip_not_compile_gpu
@pytest.mark.api_nn_fused_feedforward_parameters
def test_fused_feedforward5():
    """
    activation=gelu
    """
    obj.delta = 1e-2
    x = np.random.rand(1, 2, 2)
    w1 = np.random.rand(2, 2)
    w2 = np.random.rand(2, 2)
    res = cal_fused_feedforward(x, w1, w2, activation="gelu", dropout1_rate=0, dropout2_rate=0)
    obj.run(res=res, x=x, linear1_weight=w1, linear2_weight=w2, activation="gelu", dropout1_rate=0, dropout2_rate=0)
