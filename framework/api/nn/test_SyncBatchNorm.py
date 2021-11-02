#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_SyncBatchNorm
"""

import paddle
import pytest
import numpy as np
from paddle.nn import SyncBatchNorm
from apibase import compare


@pytest.mark.api_nn_SyncBatchNorm_parameters
def test_dygraph1():
    """
    test num_features=2
    """
    if paddle.is_compiled_with_cuda():

        x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype("float32")
        x = paddle.to_tensor(x)
        sync_batch_norm = SyncBatchNorm(num_features=2)
        res = sync_batch_norm(x)
        expect = np.array(
            [[[[0.26824948, 1.0936325], [0.26824948, -1.6301316]], [[0.8095662, -0.665287], [-1.2744656, 1.1301866]]]]
        )
        compare(res.numpy(), expect)


@pytest.mark.api_nn_SyncBatchNorm_parameters
def test_dygraph2():
    """
    test epsilon=1e-4
    """
    if paddle.is_compiled_with_cuda():
        x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype("float32")
        x = paddle.to_tensor(x)
        sync_batch_norm = SyncBatchNorm(num_features=2, epsilon=1e-4, weight_attr=False, bias_attr=False)
        res = sync_batch_norm(x)
        expect = np.array(
            [[[[0.26743087, 1.0902951], [0.26743087, -1.625157]], [[0.809192, -0.6649795], [-1.2738764, 1.1296642]]]]
        )
        compare(res.numpy(), expect)


@pytest.mark.api_nn_SyncBatchNorm_parameters
def test_dygraph3():
    """
    test weight_attr, epsilon=0.001
    """
    if paddle.is_compiled_with_cuda():
        np.random.seed(33)
        x = np.random.random(size=(2, 1, 2, 3)).astype("float32")
        x = paddle.to_tensor(x)
        w_param_attrs = paddle.ParamAttr(
            name="fc_weight", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
        )
        sync_batch_norm = SyncBatchNorm(num_features=1, epsilon=0.001, weight_attr=w_param_attrs, bias_attr=None)
        res = sync_batch_norm(x)
        expect = np.array(
            [
                [[[-0.8187444, -0.14580934, -0.27619296], [-0.7793649, 1.2584797, -1.0307478]]],
                [[[-1.5831456, 1.5352368, 0.6240252], [-0.02351551, 1.574567, -0.334787]]],
            ]
        )
        compare(res.numpy(), expect)


@pytest.mark.api_nn_SyncBatchNorm_parameters
def test_dygraph4():
    """
    test bias_attr,data_format='NCHW'
    """
    if paddle.is_compiled_with_cuda():

        x = np.arange(12).reshape(2, 2, 3).astype("float32")
        x = paddle.to_tensor(x)
        p_bias_attr = paddle.ParamAttr(
            name="fc_bias", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
        )
        sync_batch_norm = SyncBatchNorm(
            num_features=2, epsilon=1e-4, weight_attr=False, bias_attr=p_bias_attr, data_format="NCHW", name=None
        )
        res = sync_batch_norm(x)
        expect = np.array(
            [
                [[-1.2865285, -0.9648963, -0.64326423], [-1.2865283, -0.9648962, -0.6432642]],
                [[0.64326423, 0.9648963, 1.2865285], [0.6432642, 0.9648962, 1.2865283]],
            ]
        )
        compare(res.numpy(), expect)


@pytest.mark.api_nn_SyncBatchNorm_parameters
def test_dygraph5():
    """
    data_format='NCHW',weight_attr,bias_attr
    """
    if paddle.is_compiled_with_cuda():

        x = np.arange(8).reshape((2, 2, 2)).astype("float32")
        x = paddle.to_tensor(x)
        p_bias_attr = paddle.ParamAttr(
            name="fc_bias1", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
        )
        p_weight_attr = paddle.ParamAttr(
            name="fc_weight1", learning_rate=0.5, regularizer=paddle.regularizer.L2Decay(1.0), trainable=True
        )
        sync_batch_norm = SyncBatchNorm(
            num_features=2,
            epsilon=1e-4,
            weight_attr=p_weight_attr,
            bias_attr=p_bias_attr,
            data_format="NCHW",
            name=None,
        )
        res = sync_batch_norm(x)
        expect = np.array(
            [[[-1.2126639, -0.7275983], [-1.2126639, -0.7275983]], [[0.7275983, 1.2126639], [0.7275983, 1.2126639]]]
        )
        compare(res.numpy(), expect)


@pytest.mark.api_nn_SyncBatchNorm_parameters
def test_dygraph6():
    """
    data_format=NHWC
    """
    if paddle.is_compiled_with_cuda():

        np.random.seed(33)
        x = np.random.random(size=(3, 1, 2, 1)).astype("float32")
        x = paddle.to_tensor(x)
        sync_batch_norm = SyncBatchNorm(
            num_features=1, epsilon=1e-4, weight_attr=None, bias_attr=None, data_format="NHWC", name=None
        )
        res = sync_batch_norm(x)
        expect = np.array(
            [[[[-0.6815632], [0.20042753]]], [[[0.02953863], [-0.6299499]]], [[[2.0409765], [-0.9594281]]]]
        )
        compare(res.numpy(), expect)
