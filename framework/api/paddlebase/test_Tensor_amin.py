#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_amin
"""

from apibase import randtool
import paddle
import pytest
import numpy as np


def cal_amin(x, jud=True, axis=None, keepdim=False):
    """
    calculate amin
    """
    res = np.amin(x, axis=axis, keepdims=keepdim)
    if axis is None and keepdim is False:
        res = np.array([res])
    grad = np.where(x == res, 1, 0)
    grad = grad / np.sum(grad, axis=axis, keepdims=True)
    return res, grad if jud else res


def cal_api(x, dtype="float32", axis=None, keepdim=False):
    """
    calculate api
    """
    x = x.astype(dtype)
    dynamic_grad = None
    xp = paddle.to_tensor(x, stop_gradient=False, dtype=dtype)
    dynamic_res = paddle.Tensor.amin(xp, axis=axis, keepdim=keepdim)
    if dtype == "float32" or dtype == "float64":
        dynamic_res.backward()
        dynamic_grad = xp.grad.numpy()

    paddle.enable_static()
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
            data0 = paddle.static.data(name="s0", shape=x.shape, dtype=dtype)
            feed = {"s0": x}
            out = paddle.Tensor.amin(data0, axis=axis, keepdim=keepdim)
            if dtype == "float32" or dtype == "float64":
                data0.stop_gradient = False
                grad = paddle.static.gradients(out, data0)
            exe = paddle.static.Executor()
            exe.run(startup_program)
            if dtype == "float32" or dtype == "float64":
                static_res = exe.run(main_program, feed=feed, fetch_list=[out] + grad)
            else:
                static_res = exe.run(main_program, feed=feed, fetch_list=[out])
    paddle.disable_static()
    assert np.allclose(dynamic_res.numpy(), static_res[0])
    if dtype == "float32" or dtype == "float64":
        assert np.allclose(dynamic_grad, static_res[1])
    return dynamic_res.numpy(), dynamic_grad


@pytest.mark.api_base_amin_vartype
def test_amin_base():
    """
    base
    """
    x = randtool("int", -4, 40, (10,))
    types0 = ["int32", "int64"]
    types1 = ["float32", "float64"]
    for dtype in types0:
        np_res = cal_amin(x, jud=False)
        res, grad = cal_api(x, dtype=dtype)
        assert np.allclose(np_res, res)

    for dtype in types1:
        np_res, np_grad = cal_amin(x)
        res, grad = cal_api(x, dtype=dtype)
        assert np.allclose(np_res, res)
        assert np.allclose(np_grad, grad)


@pytest.mark.api_base_amin_parameters
def test_amin0():
    """
    default
    """
    x = randtool("float", -4, 14, (2, 4))
    np_res, np_grad = cal_amin(x)
    res, grad = cal_api(x)
    assert np.allclose(np_res, res)
    assert np.allclose(grad, np_grad)


@pytest.mark.api_base_amin_parameters
def test_amin1():
    """
    x: 3d-tensor
    """
    x = randtool("float", -4, 14, (2, 5, 4))
    np_res, np_grad = cal_amin(x)
    res, grad = cal_api(x)
    assert np.allclose(np_res, res)
    assert np.allclose(grad, np_grad)


@pytest.mark.api_base_amin_parameters
def test_amin2():
    """
    x: 4d-tensor
    """
    x = randtool("float", -40, 140, (3, 2, 5, 4))
    np_res, np_grad = cal_amin(x)
    res, grad = cal_api(x)
    assert np.allclose(np_res, res)
    assert np.allclose(grad, np_grad)


@pytest.mark.api_base_amin_parameters
def test_amin3():
    """
    x: 4d-tensor
    axis=-1
    keepdim=True
    """
    x = randtool("float", -40, 140, (3, 2, 4, 5))
    np_res, np_grad = cal_amin(x, axis=-1, keepdim=True)
    res, grad = cal_api(x, axis=-1, keepdim=True)
    assert np.allclose(np_res, res)
    assert np.allclose(grad, np_grad)


@pytest.mark.api_base_amin_parameters
def test_amin4():
    """
    x: 4d-tensor
    axis=2
    keepdim=True
    """
    x = randtool("float", -40, 140, (3, 2, 5, 4))
    np_res, np_grad = cal_amin(x, axis=2, keepdim=True)
    res, grad = cal_api(x, axis=2, keepdim=True)
    assert np.allclose(np_res, res)
    assert np.allclose(grad, np_grad)


@pytest.mark.api_base_amin_parameters
def test_amin5():
    """
    x: 4d-tensor
    axis=(1, 2)
    keepdim=True
    """
    x = randtool("float", -40, 140, (3, 2, 5, 4))
    np_res, np_grad = cal_amin(x, axis=(1, 2), keepdim=True)
    res, grad = cal_api(x, axis=(1, 2), keepdim=True)
    assert np.allclose(np_res, res)
    assert np.allclose(grad, np_grad)
