#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.margin_cross_entropy
"""
import sys
import paddle
import pytest
from runner import compare

sys.path.append("../..")
from utils.interceptor import skip_not_compile_gpu


@skip_not_compile_gpu
@pytest.mark.loss_margin_cross_entropy_parameters
def test_margin_cross_entropy_3():
    """
    test nn.functional.margin_cross_entropy test 3 softmax
    """
    seed = 33
    paddle.seed(seed)
    m1 = 1.0
    m2 = 0.7
    m3 = 0.2
    s = 64.0
    batch_size = 2
    feature_length = 4
    num_classes = 4
    label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype="int64")
    X = paddle.randn(shape=[batch_size, feature_length], dtype="float64")
    X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
    X = paddle.divide(X, X_l2)
    W = paddle.randn(shape=[feature_length, num_classes], dtype="float64")
    W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
    W = paddle.divide(W, W_l2)
    logits = paddle.matmul(X, W)
    loss, softmax = paddle.nn.functional.margin_cross_entropy(
        logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None
    )

    # add arcface margin to logit
    theta = paddle.acos(logits)
    one_hot_label = paddle.nn.functional.one_hot(label, num_classes=num_classes)
    if m1 != 1.0:
        theta = m1 * theta
    if m2 != 0.0:
        theta = theta + m2
    margin_cos = paddle.cos(theta)
    if 3 != 0.0:
        margin_cos = margin_cos - m3
    diff = one_hot_label * (margin_cos - logits)
    arc_data = (logits + diff) * s

    loss_b, softmax_b = paddle.nn.functional.softmax_with_cross_entropy(
        logits=arc_data, label=paddle.reshape(label, (-1, 1)), return_softmax=True
    )

    compare(softmax.numpy(), softmax_b.numpy())
