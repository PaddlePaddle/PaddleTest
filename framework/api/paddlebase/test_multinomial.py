#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test multinomial
"""
import paddle
import pytest
import numpy as np
from paddle import fluid


if fluid.is_compiled_with_cuda() is True:
    devices = ["gpu", "cpu"]
else:
    devices = ["cpu"]
if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    places = [fluid.CPUPlace()]
dtypes = [np.float32, np.float64]
value_dtypes = [np.int16, np.int32, np.int64]

seed = 33


@pytest.mark.api_base_multinomial_vartype
def test_multinomial_dygraph1():
    """
    x=1-D Tensor,num_samples=1
    """
    paddle.disable_static()
    for device in devices:
        for t in dtypes:
            np.random.seed(seed)
            paddle.set_device(device)
            x = np.random.random(6).astype(t)
            res = paddle.multinomial(paddle.to_tensor(x), name=None)
            assert np.allclose(res.shape, [1], atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_multinomial_parameters
def test_multinomial_dygraph2():
    """
    x=1-D Tensor,num_samples=5,replacement=True
    """
    paddle.disable_static()
    for device in devices:
        for t in dtypes:
            np.random.seed(seed)
            paddle.set_device(device)
            x = np.random.random(7).astype(t)
            res = paddle.multinomial(paddle.to_tensor(x), num_samples=4, name=None)
            assert np.allclose(res.shape, [4], atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_multinomial_parameters
def test_multinomial_dygraph3():
    """
    x=2-D Tensor,num_samples=5,replacement=True
    """
    paddle.disable_static()
    for device in devices:
        for t in dtypes:
            np.random.seed(seed)
            paddle.set_device(device)
            x = np.random.random([2, 7]).astype(t)
            res = paddle.multinomial(paddle.to_tensor(x), num_samples=5, replacement=True, name=None)
            assert np.allclose(res.shape, [2, 5], atol=0.005, rtol=0.05, equal_nan=True)
    paddle.enable_static()


@pytest.mark.api_base_multinomial_vartype
def test_multinomial_static1():
    """
    x=1-D Tensor,num_samples=1
    :return:
    """
    paddle.enable_static()
    for device in devices:
        for t in dtypes:
            np.random.seed(seed)
            paddle.set_device(device)
            main_program = fluid.Program()
            startup_program = fluid.Program()
            x = np.random.random([6]).astype(t)
            feed = {"x": x}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    x = paddle.static.data(name="x", shape=x.shape, dtype=t)
                    res = paddle.multinomial(x=x, name=None)
                    exe = fluid.Executor()
                    exe.run(startup_program)
                    static_out = exe.run(main_program, feed=feed, fetch_list=[res])
                    assert np.allclose(np.array(static_out).shape, [1], atol=0.005, rtol=0.05, equal_nan=True)


@pytest.mark.api_base_multinomial_parameters
def test_multinomial_static2():
    """
    x=1-D Tensor,num_samples=8,replacement=True
    :return:
    """
    paddle.enable_static()
    for device in devices:
        for t in dtypes:
            np.random.seed(seed)
            paddle.set_device(device)
            main_program = fluid.Program()
            startup_program = fluid.Program()
            x = np.random.random([5]).astype(t)
            feed = {"x": x}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    x = paddle.static.data(name="x", shape=x.shape, dtype=t)
                    res = paddle.multinomial(x=x, num_samples=8, replacement=True, name=None)
                    exe = fluid.Executor()
                    exe.run(startup_program)
                    static_out = exe.run(main_program, feed=feed, fetch_list=[res])
                    assert np.allclose(np.array(static_out).shape, [1, 8], atol=0.005, rtol=0.05, equal_nan=True)


@pytest.mark.api_base_multinomial_parameters
def test_multinomial_static3():
    """
    x=2-D Tensor,num_samples=5,replacement=True
    :return:
    """
    paddle.enable_static()
    for device in devices:
        for t in dtypes:
            np.random.seed(seed)
            paddle.set_device(device)
            main_program = fluid.Program()
            startup_program = fluid.Program()
            x = np.random.random([6, 1]).astype(t)
            feed = {"x": x}
            with fluid.unique_name.guard():
                with fluid.program_guard(main_program=main_program, startup_program=startup_program):
                    x = paddle.static.data(name="x", shape=x.shape, dtype=t)
                    res = paddle.multinomial(x=x, num_samples=5, replacement=True, name=None)
                    exe = fluid.Executor()
                    exe.run(startup_program)
                    static_out = exe.run(main_program, feed=feed, fetch_list=[res])
                    assert np.allclose(np.array(static_out).shape, [1, 6, 5], atol=0.005, rtol=0.05, equal_nan=True)
