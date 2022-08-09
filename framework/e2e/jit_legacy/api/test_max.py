#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test max
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_max_vartype
def test_jit_max_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.max(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.max
        """
        return paddle.max(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="max_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_max_vartype
def test_jit_max_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.max(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.max
        """
        a = paddle.max(inputs, axis=0)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="max_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_max_parameters
def test_jit_max_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.max(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.max
        """
        return paddle.max(inputs, axis=-3)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="max_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_max_parameters
def test_jit_max_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.max(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.max
        """
        return paddle.max(inputs, axis=-3, keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="max_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_max_parameters
def test_jit_max_4():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.max(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.max
        """
        return paddle.max(inputs, axis=[-2, 1], keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="max_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


# @pytest.mark.jit_max_parameters
# def test_jit_max_5():
#     """
#     @paddle.jit.to_static
#     def fun(inputs):
#         return paddle.max(inputs)
#     inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
#     dtype=["float32"]
#     """
#
#     @paddle.jit.to_static
#     def func(inputs):
#         """
#         paddle.max
#         """
#         return paddle.max(inputs, axis=-1)
#
#     inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
#     runner = Runner(func=func, name="max_2", dtype=["float32"], ftype="func")
#     runner.add_kwargs_to_dict("params_group1", inputs=inps)
#     runner.run()
