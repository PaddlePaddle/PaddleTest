#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_weight_norm
"""
import paddle
import pytest
import numpy as np


def cal_weight_norm(weight, dim=0):
    """
    calculate weight_norm api
    """
    shape = weight.shape
    g_dim = list(range(len(shape)))
    g_dim.pop(dim)
    convert_dim = [dim] + g_dim
    matrix = np.transpose(weight, convert_dim)
    weight_g = []
    weight_v = weight
    for item in matrix:
        weight_g.append(np.linalg.norm(item))
    return np.array(weight_g), weight_v


def cal_static(layer, dim=0):
    """
    calculate static result
    """
    paddle.enable_static()
    main_program, startup_program = paddle.static.default_main_program(), paddle.static.default_startup_program()
    with paddle.static.program_guard(startup_program=startup_program, main_program=main_program):
        paddle.nn.utils.weight_norm(layer, dim=dim)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res1, res2 = exe.run(main_program, fetch_list=[layer.weight_g, layer.weight_v])
        return res1, res2


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm0():
    """
    layer: conv2d
    """
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv2d.weight)

    paddle.nn.utils.weight_norm(conv2d)
    dynamic_res1, dynamic_res2 = conv2d.weight_g, conv2d.weight_v

    paddle.enable_static()
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv2d)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm1():
    """
    layer: conv2d
    dim = 1
    """
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv2d.weight, dim=1)

    paddle.nn.utils.weight_norm(conv2d, dim=1)
    dynamic_res1, dynamic_res2 = conv2d.weight_g, conv2d.weight_v

    paddle.enable_static()
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv2d, dim=1)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm2():
    """
    layer: conv2d
    dim = 2
    """
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv2d.weight, dim=2)

    paddle.nn.utils.weight_norm(conv2d, dim=2)
    dynamic_res1, dynamic_res2 = conv2d.weight_g, conv2d.weight_v

    paddle.enable_static()
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv2d, dim=2)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm3():
    """
    layer: conv2d
    dim = 3
    """
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv2d.weight, dim=3)

    paddle.nn.utils.weight_norm(conv2d, dim=3)
    dynamic_res1, dynamic_res2 = conv2d.weight_g, conv2d.weight_v

    paddle.enable_static()
    conv2d = paddle.nn.Conv2D(3, 5, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv2d, dim=3)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm4():
    """
    layer: conv1d
    """
    conv1d = paddle.nn.Conv1D(4, 5, 2, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv1d.weight)

    paddle.nn.utils.weight_norm(conv1d)
    dynamic_res1, dynamic_res2 = conv1d.weight_g, conv1d.weight_v

    paddle.enable_static()
    conv1d = paddle.nn.Conv1D(4, 5, 2, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv1d)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm5():
    """
    layer: conv1d
    dim = 1
    """
    conv1d = paddle.nn.Conv1D(4, 5, 4, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv1d.weight, dim=1)

    paddle.nn.utils.weight_norm(conv1d, dim=1)
    dynamic_res1, dynamic_res2 = conv1d.weight_g, conv1d.weight_v

    paddle.enable_static()
    conv1d = paddle.nn.Conv1D(4, 5, 4, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv1d, dim=1)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm6():
    """
    layer: conv1d
    dim = 2
    """
    conv1d = paddle.nn.Conv1D(4, 15, 4, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv1d.weight, dim=2)

    paddle.nn.utils.weight_norm(conv1d, dim=2)
    dynamic_res1, dynamic_res2 = conv1d.weight_g, conv1d.weight_v

    paddle.enable_static()
    conv1d = paddle.nn.Conv1D(4, 15, 4, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv1d, dim=2)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm7():
    """
    layer: conv3d
    """
    conv3d = paddle.nn.Conv3D(4, 5, 4, 4, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv3d.weight)

    paddle.nn.utils.weight_norm(conv3d)
    dynamic_res1, dynamic_res2 = conv3d.weight_g, conv3d.weight_v

    paddle.enable_static()
    conv3d = paddle.nn.Conv3D(4, 5, 4, 4, 3, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv3d)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm8():
    """
    layer: conv3d
    dim = 1
    """
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv3d.weight, dim=1)

    paddle.nn.utils.weight_norm(conv3d, dim=1)
    dynamic_res1, dynamic_res2 = conv3d.weight_g, conv3d.weight_v

    paddle.enable_static()
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv3d, dim=1)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm9():
    """
    layer: conv3d
    dim = 2
    """
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv3d.weight, dim=2)

    paddle.nn.utils.weight_norm(conv3d, dim=2)
    dynamic_res1, dynamic_res2 = conv3d.weight_g, conv3d.weight_v

    paddle.enable_static()
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv3d, dim=2)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm10():
    """
    layer: conv3d
    dim = 3
    """
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv3d.weight, dim=3)

    paddle.nn.utils.weight_norm(conv3d, dim=3)
    dynamic_res1, dynamic_res2 = conv3d.weight_g, conv3d.weight_v

    paddle.enable_static()
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv3d, dim=3)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm11():
    """
    layer: conv3d
    dim = 4
    """
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(conv3d.weight, dim=4)

    paddle.nn.utils.weight_norm(conv3d, dim=4)
    dynamic_res1, dynamic_res2 = conv3d.weight_g, conv3d.weight_v

    paddle.enable_static()
    conv3d = paddle.nn.Conv3D(4, 5, 6, 7, 8, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(conv3d, dim=4)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm12():
    """
    layer: linear
    """
    linear = paddle.nn.Linear(4, 7, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(linear.weight)

    paddle.nn.utils.weight_norm(linear)
    dynamic_res1, dynamic_res2 = linear.weight_g, linear.weight_v

    paddle.enable_static()
    linear = paddle.nn.Linear(4, 7, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(linear)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)


@pytest.mark.api_nn_weight_norm_parameters
def test_weight_norm13():
    """
    layer: linear
    dim=1
    """
    linear = paddle.nn.Linear(4, 7, weight_attr=paddle.nn.initializer.Constant(4.0))
    res1, res2 = cal_weight_norm(linear.weight, dim=1)

    paddle.nn.utils.weight_norm(linear, dim=1)
    dynamic_res1, dynamic_res2 = linear.weight_g, linear.weight_v

    paddle.enable_static()
    linear = paddle.nn.Linear(4, 7, weight_attr=paddle.nn.initializer.Constant(4.0))
    static_res1, static_res2 = cal_static(linear, dim=1)
    paddle.disable_static()

    assert np.allclose(dynamic_res1.numpy(), static_res1)
    assert np.allclose(dynamic_res2.numpy(), static_res2)
    assert np.allclose(dynamic_res1.numpy(), res1)
    assert np.allclose(static_res2, res2)
