#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test mean
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_mean_vartype
def test_jit_mean_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.mean(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.mean
        """
        return paddle.mean(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="mean_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_mean_vartype
def test_jit_mean_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.mean(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.mean
        """
        a = paddle.mean(inputs, axis=0)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="mean_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_mean_parameters
def test_jit_mean_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.mean(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.mean
        """
        return paddle.mean(inputs, axis=-3)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="mean_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_mean_parameters
def test_jit_mean_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.mean(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.mean
        """
        return paddle.mean(inputs, axis=-1, keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="mean_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_mean_parameters
def test_jit_mean_4():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.mean(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.mean
        """
        return paddle.mean(inputs, axis=[1, 3], keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="mean_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


# @pytest.mark.jit_mean_parameters
# def test_jit_mean_5():
#     """
#     @paddle.jit.to_static
#     def fun(inputs):
#         return paddle.mean(inputs)
#     inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
#     dtype=["float32"]
#     """
#
#     @paddle.jit.to_static
#     def func(inputs):
#         """
#         paddle.mean
#         """
#         return paddle.mean(inputs, axis=-1)
#
#     inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
#     runner = Runner(func=func, name="mean_2", dtype=["float32"], ftype="func")
#     runner.add_kwargs_to_dict("params_group1", inputs=inps)
#     runner.run()
