#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test full
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_full_vartype
def test_jit_full_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.full(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.full
        """
        a = paddle.full(inputs, fill_value=-2.5)
        return a

    inps = np.array([3, 2, 5])
    runner = Runner(func=func, name="full_1", dtype=["int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_full_parameters
def test_jit_full_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.full(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.full
        """
        return paddle.full(inputs, fill_value=paddle.to_tensor([4]))

    inps = np.array([3, 2, 5])
    runner = Runner(func=func, name="full_2", dtype=["int32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
