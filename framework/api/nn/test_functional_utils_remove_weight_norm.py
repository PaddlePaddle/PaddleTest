#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_remove_weight_norm
"""
import paddle
import pytest


@pytest.mark.api_nn_remove_weight_norm_parameters
def test_remove_weight_norm0():
    """
    layer: conv2d
    """
    with pytest.raises(AttributeError) as att_info:
        conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
        paddle.nn.utils.weight_norm(conv2d)
        paddle.nn.utils.remove_weight_norm(conv2d)
        res = conv2d.weight_g
        assert att_info.type == AttributeError
        assert res is None


@pytest.mark.api_nn_remove_weight_norm_parameters
def test_remove_weight_norm1():
    """
    layer: conv3d
    """
    with pytest.raises(AttributeError) as att_info:
        conv3d = paddle.nn.Conv3D(3, 5, 3, 4, 7, weight_attr=paddle.nn.initializer.Constant(4.0))
        paddle.nn.utils.weight_norm(conv3d)
        paddle.nn.utils.remove_weight_norm(conv3d)
        res = conv3d.weight_g
        assert att_info.type == AttributeError
        assert res is None


@pytest.mark.api_nn_remove_weight_norm_parameters
def test_remove_weight_norm2():
    """
    layer: linear
    """
    with pytest.raises(AttributeError) as att_info:
        linear = paddle.nn.Linear(3, 5, weight_attr=paddle.nn.initializer.Constant(4.0))
        paddle.nn.utils.weight_norm(linear)
        paddle.nn.utils.remove_weight_norm(linear)
        res = linear.weight_g
        assert att_info.type == AttributeError
        assert res is None
