#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test empty_like
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_empty_like_vartype
def test_jit_empty_like_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.empty_like(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.empty_like
        """
        return paddle.empty_like(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="empty_like_base", dtype=["float32", "float64", "int32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_empty_like_parameters
def test_jit_empty_like_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.empty_like(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.empty_like
        """
        return paddle.empty_like(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="empty_like_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
