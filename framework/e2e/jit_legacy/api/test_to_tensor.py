#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test to_tensor
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_to_tensor_vartype
def test_jit_to_tensor_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.to_tensor(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "uint8"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.to_tensor
        """
        return paddle.to_tensor(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(
        func=func, name="to_tensor_base", dtype=["float32", "float64", "int32", "int64", "uint8"], ftype="func"
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_to_tensor_vartype
def test_jit_to_tensor_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.to_tensor(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "uint8"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.to_tensor
        """
        a = paddle.to_tensor(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(
        func=func, name="to_tensor_1", dtype=["float32", "float64", "int32", "int64", "uint8"], ftype="func"
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_to_tensor_parameters
def test_jit_to_tensor_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.to_tensor(inputs)
    inputs=randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.to_tensor
        """
        return paddle.to_tensor(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="to_tensor_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_to_tensor_parameters
def test_jit_to_tensor_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.to_tensor(inputs)
    inputs=randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float16"]
    only support gpu
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.to_tensor
        """
        return paddle.to_tensor(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="to_tensor_3", dtype=["float16"], ftype="func")
    runner.places = ["gpu:0"]
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    if paddle.device.is_compiled_with_cuda() is True:
        runner.run()
