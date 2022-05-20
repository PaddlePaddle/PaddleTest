#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test is_tensor
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_is_tensor_vartype
def test_jit_is_tensor_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.is_tensor(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.is_tensor
        """
        return paddle.is_tensor(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="is_tensor_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_is_tensor_parameters
def test_jit_is_tensor_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.is_tensor(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.is_tensor
        """
        return paddle.is_tensor(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="is_tensor_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_is_tensor_parameters
def test_jit_is_tensor_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.is_tensor(inputs)
    inputs=[]
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.is_tensor
        """
        return paddle.is_tensor(inputs)

    inps = np.array([])
    runner = Runner(func=func, name="is_tensor_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_is_tensor_parameters
def test_jit_is_tensor_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.is_tensor(inputs)
    inputs=list
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.is_tensor
        """
        return paddle.is_tensor(inputs)

    inps = [1, 0, 2, 1]
    runner = Runner(func=func, name="is_tensor_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
