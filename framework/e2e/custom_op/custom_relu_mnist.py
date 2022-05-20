#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
mnist
"""
from __future__ import print_function
import os
import numpy as np
import paddle
from paddle.utils.cpp_extension import load


current_path = os.path.dirname(os.path.abspath(__file__))


custom_ops = load(name="relu_op_jit", sources=[current_path + "/relu_op.cc"])


class SimpleImgConvPool(paddle.nn.Layer):
    """
    SimpleImgConvPool
    """

    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        pool_size,
        pool_stride,
        pool_padding=0,
        pool_type="max",
        global_pooling=False,
        conv_stride=1,
        conv_padding=0,
        conv_dilation=1,
        conv_groups=1,
        act=None,
        use_cudnn=False,
        param_attr=None,
        bias_attr=None,
    ):
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            weight_attr=None,
            bias_attr=None,
        )
        self._act = act

        self._pool2d = paddle.fluid.dygraph.nn.Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn,
        )

    def forward(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        x = self._conv2d(inputs)
        x = getattr(paddle.nn.functional, self._act)(x) if self._act else x
        x = self._pool2d(x)
        return x


class CUSTOMMNIST(paddle.nn.Layer):
    """
    CUSTOM_MNIST
    """

    def __init__(self):
        super(CUSTOMMNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(1, 20, 5, 2, 2)

        self._simple_img_conv_pool_2 = SimpleImgConvPool(20, 50, 5, 2, 2)

        self.pool_2_shape = 50 * 4 * 4
        self._fc = paddle.nn.Linear(
            in_features=self.pool_2_shape,
            out_features=10,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal()),
        )

    def forward(self, inputs, label=None):
        """

        Args:
            inputs:
            label:

        Returns:

        """
        x = self._simple_img_conv_pool_1(inputs)
        x = custom_ops.custom_relu(x)
        x = self._simple_img_conv_pool_2(x)
        x = custom_ops.custom_relu(x)
        x = paddle.fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        x = paddle.nn.functional.softmax(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


def reader_decorator(reader):
    """

    Args:
        reader:

    Returns:

    """

    def __reader__():
        """

        Returns:

        """
        for item in reader():
            img = np.array(item[0]).astype("float32").reshape(1, 28, 28)
            label = np.array(item[1]).astype("int64").reshape(1)
            yield img, label

    return __reader__
