#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_grad
"""
import pytest
import paddle
import paddle.fluid as fluid
import numpy as np

# from apibase import randtool
# from apibase import compare


# global params
types = [np.float32, np.float64]
if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    # default
    places = [fluid.CPUPlace()]
seed = 33


@pytest.mark.api_base_grad_parameters
def test_dygraph():
    """
    test_dygraph
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            # your own test code
            x = np.array([8]).astype(t)
            y = np.array([2]).astype(t)
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            x.stop_gradient = False
            y.stop_gradient = False
            z = x * y
            res = paddle.grad(outputs=z, inputs=x, retain_graph=False, create_graph=False)
            assert res[0].numpy() == 2
            paddle.enable_static()


@pytest.mark.api_base_grad_parameters
def test_dygraph1():
    """
    z = x * x * x
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            # your own test code
            x = np.array([8]).astype(t)
            y = np.array([2]).astype(t)
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            x.stop_gradient = False
            y.stop_gradient = False
            z = x * x * x
            # res = paddle.grad(outputs=z, inputs=x, retain_graph=False, create_graph=False)
            # assert res[0].numpy() == 3 * 8 * 8
            res = paddle.grad(outputs=z, inputs=x, retain_graph=True, create_graph=True)
            assert res[0].numpy() == 3 * 8 * 8
            paddle.enable_static()


@pytest.mark.api_base_grad_parameters
def test_dygraph2():
    """
    z = x * x * x
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            # your own test code
            x = np.array([8]).astype(t)
            y = np.array([2]).astype(t)
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            x.stop_gradient = False
            y.stop_gradient = False
            z = x * x * x
            # res = paddle.grad(outputs=z, inputs=x, retain_graph=False, create_graph=False)
            # assert res[0].numpy() == 3 * 8 * 8
            res, = paddle.grad(outputs=z, inputs=x, retain_graph=True, create_graph=True)
            res.backward()
            assert x.gradient() == 6 * 8
            paddle.enable_static()


@pytest.mark.api_base_grad_parameters
def test_dygraph3():
    """
    z = x * x * x
    """
    for place in places:
        for t in types:
            paddle.disable_static(place)
            # your own test code
            x = np.array([8]).astype(t)
            x1 = x
            y = np.array([2]).astype(t)
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            x.stop_gradient = False
            y.stop_gradient = False
            z = x * x * x
            # res = paddle.grad(outputs=z, inputs=x, retain_graph=False, create_graph=False)
            # assert res[0].numpy() == 3 * 8 * 8
            res, = paddle.grad(outputs=z, inputs=x, retain_graph=True, create_graph=True)
            o = z + res
            o.backward()
            assert x.gradient() == 3 * x1 * x1 + 6 * x
            paddle.enable_static()
