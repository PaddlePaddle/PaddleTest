#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test var
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_var_vartype
def test_jit_var_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.var(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.var
        """
        return paddle.var(inputs)

    inps = np.array([[1.5, -2.1, 3.2, 0], [1.5, -2.1, 3.2, 0]])
    runner = Runner(func=func, name="var_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_var_vartype
def test_jit_var_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.var(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.var
        """
        a = paddle.var(inputs, axis=-1, unbiased=False, keepdim=False)
        return a

    inps = np.array([[1.5, -2.1, 3.2, 0], [1.5, -2.1, 3.2, 0]])
    runner = Runner(func=func, name="var_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_var_parameters
def test_jit_var_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.var(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.var
        """
        return paddle.var(inputs, axis=4, unbiased=True, keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="var_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


# @pytest.mark.jit_var_parameters
# def test_jit_var_2():
#     """
#     @paddle.jit.to_static
#     def fun(inputs):
#         return paddle.var(inputs)
#     inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
#     dtype=["float32"]
#     """
#
#     @paddle.jit.to_static
#     def func(inputs):
#         """
#         paddle.var
#         """
#         return paddle.var(inputs, axis=-1, unbiased=True, keepdim=False)
#
#     inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
#     runner = Runner(func=func, name="var_2", dtype=["float32"], ftype="func")
#     runner.add_kwargs_to_dict("params_group1", inputs=inps)
#     runner.run()
