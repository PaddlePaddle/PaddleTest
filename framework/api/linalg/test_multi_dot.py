#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_multi_dot
"""

import copy
import logging
import paddle
import numpy as np
import pytest


# place = []
if paddle.device.is_compiled_with_cuda() is True:
    paddle.device.set_device("gpu:0")
    place = paddle.CUDAPlace(0)
else:
    paddle.device.set_device("cpu")
    place = paddle.CPUPlace()


def cal_multi_dot(dtype, **kwargs):
    """
    calculate multi_dot api
    """

    x = {}
    for k, v in kwargs.items():
        x[k] = paddle.to_tensor(v, stop_gradient=False)
    dynamic_res = paddle.linalg.multi_dot(list(x.values()))

    loss = paddle.mean(dynamic_res)
    loss.backward()
    dynamic_grad = {}
    for k in list(kwargs.keys()):
        dynamic_grad[k] = x[k].grad

    paddle.enable_static()
    feed = {}
    for k, v in kwargs.items():
        feed[k] = v.astype(dtype)
    params = copy.deepcopy(kwargs)
    main_program, startup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
        for k, v in params.items():
            params[k] = paddle.static.data(name=k, shape=v.shape, dtype=dtype)
            params[k].stop_gradient = False
        x = list(params.values())
        y = paddle.linalg.multi_dot(x)

        loss = paddle.mean(y)
        static_grad = {}
        for k in list(kwargs.keys()):
            static_grad[k] = paddle.static.gradients(loss, params[k])
        logging.info(place)
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        # print(list(static_grad.values()))
        # print([output] + list(grad_var.values()))
        static_res = exe.run(main_program, feed=feed, fetch_list=[y] + list(static_grad.values()), return_numpy=True)
        paddle.disable_static()
        assert np.allclose(dynamic_res.numpy(), static_res[0])
        for i in range(len(kwargs)):
            assert np.allclose(list(dynamic_grad.values())[i].numpy(), static_res[i + 1])

    return dynamic_res.numpy(), list(dynamic_grad.values())


def cal_loss(**kwargs):
    """
    calculate loss
    """
    x = {}
    for k, v in kwargs.items():
        x[k] = paddle.to_tensor(v, stop_gradient=False)
    res = paddle.linalg.multi_dot(list(x.values()))
    loss = paddle.mean(res)
    return loss


def numerical_grad(**kwargs):
    """
    calculate numerical gradient
    """
    gap = 1e-7
    loss = cal_loss(**kwargs)
    numeric_grad = {}
    for k, v in kwargs.items():
        grad = []
        shape = v.shape
        for i in range(len(v.flatten())):
            tmp = v.flatten()
            tmp[i] = tmp[i] + gap
            tmp = tmp.reshape(shape)
            # print(tmp)
            kwargs[k] = tmp
            loss_delta = cal_loss(**kwargs)
            g = (loss_delta - loss) / gap
            grad.append(g[0])
            # recover v to self.kwargs
            kwargs[k] = v
        numeric_grad[k] = np.array(grad).reshape(shape)

    return list(numeric_grad.values())


@pytest.mark.api_base_multi_dot_vartype
def test_multi_dot_base():
    """
    base
    """
    types = ["float32", "float64"]
    np.random.seed(23)
    x1 = np.random.rand(2, 4)
    x2 = np.random.rand(4, 3)
    res = np.dot(x1, x2)
    for d in types:
        api_res, api_grad = cal_multi_dot(d, x1=x1, x2=x2)
        assert api_res.shape == res.shape
        assert np.allclose(api_res, res)
        numeric_grad = numerical_grad(x1=x1, x2=x2)
        for i in range(2):
            assert np.allclose(api_grad[i], numeric_grad[i])


@pytest.mark.api_base_multi_dot_parameters
def test_multi_dot0():
    """
    default
    """
    types = ["float32", "float64"]
    x1 = np.random.rand(4, 4)
    x2 = np.random.rand(4, 31)
    res = np.dot(x1, x2)
    for d in types:
        api_res, api_grad = cal_multi_dot(d, x1=x1, x2=x2)
        assert api_res.shape == res.shape
        assert np.allclose(api_res, res)
        numeric_grad = numerical_grad(x1=x1, x2=x2)
        for i in range(2):
            assert np.allclose(api_grad[i], numeric_grad[i])


@pytest.mark.api_base_multi_dot_parameters
def test_multi_dot1():
    """
    first_tensor: vector
    """
    types = ["float32", "float64"]
    x1 = np.random.rand(4)
    x2 = np.random.rand(4, 31)
    res = np.dot(x1, x2)
    for d in types:
        api_res, api_grad = cal_multi_dot(d, x1=x1, x2=x2)
        assert api_res.shape == res.shape
        assert np.allclose(api_res, res)
        numeric_grad = numerical_grad(x1=x1, x2=x2)
        for i in range(2):
            assert np.allclose(api_grad[i], numeric_grad[i])


@pytest.mark.api_base_multi_dot_parameters
def test_multi_dot2():
    """
    last_tensor: vector
    """
    types = ["float32", "float64"]
    x1 = np.random.rand(4)
    x2 = np.random.rand(4)
    res = np.dot(x1, x2).reshape([1])
    for d in types:
        api_res, api_grad = cal_multi_dot(d, x1=x1, x2=x2)
        assert api_res.shape == res.shape
        assert np.allclose(api_res, res)
        numeric_grad = numerical_grad(x1=x1, x2=x2)
        for i in range(2):
            assert np.allclose(api_grad[i], numeric_grad[i])


@pytest.mark.api_base_multi_dot_parameters
def test_multi_dot3():
    """
    multiple tensor
    """
    types = ["float32", "float64"]
    x1 = np.random.rand(4)
    x2 = np.random.rand(4, 5)
    x3 = np.random.rand(5, 2)
    x4 = np.random.rand(2)
    res = np.dot(np.dot(x1, x2), np.dot(x3, x4)).reshape([1])
    for d in types:
        api_res, api_grad = cal_multi_dot(d, x1=x1, x2=x2, x3=x3, x4=x4)
        assert api_res.shape == res.shape
        assert np.allclose(api_res, res)
        numeric_grad = numerical_grad(x1=x1, x2=x2, x3=x3, x4=x4)
        for i in range(4):
            assert np.allclose(api_grad[i], numeric_grad[i])
