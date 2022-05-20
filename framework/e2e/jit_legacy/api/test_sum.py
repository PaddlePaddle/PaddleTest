#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sum
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_sum_vartype
def test_jit_sum_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sum(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sum
        """
        return paddle.sum(inputs, axis=None, keepdim=False, dtype=None)

    inps = np.array([[1.5, 2.1, 3.2], [1.5, 2.1, 3.2]])
    runner = Runner(func=func, name="sum_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_sum_vartype
def test_jit_sum_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sum(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sum
        """
        a = paddle.sum(inputs, axis=-1, keepdim=False, dtype=None)
        return a

    inps = np.array([[1.5, 2.1, 3.2], [1.5, 2.1, 3.2]])
    runner = Runner(func=func, name="sum_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_sum_parameters
def test_jit_sum_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sum(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sum
        """
        return paddle.sum(inputs, axis=4, keepdim=True, dtype="float32")

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="sum_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_sum_parameters
def test_jit_sum_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sum(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sum
        """
        return paddle.sum(inputs, axis=1, keepdim=False, dtype="float32")

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="sum_2", dtype=["int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


# @pytest.mark.jit_sum_parameters
# def test_jit_sum_4():
#     """
#     @paddle.jit.to_static
#     def fun(inputs):
#         return paddle.sum(inputs)
#     inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
#     dtype=["float32"]
#     """
#
#     @paddle.jit.to_static
#     def func(inputs):
#         """
#         paddle.sum
#         """
#         return paddle.sum(inputs, axis=-2, keepdim=False, dtype='float32')
#
#     inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
#     runner = Runner(func=func, name="sum_2", dtype=["int64"], ftype="func")
#     runner.add_kwargs_to_dict("params_group1", inputs=inps)
#     runner.run()
