#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test all
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_all_vartype
def test_jit_all_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.all(inputs)
    inputs=np.array([True, False, True])
    dtype=["bool]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.all
        """
        return paddle.all(inputs)

    inps = np.array([True, False, True])
    runner = Runner(func=func, name="all_base", dtype=["bool"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_all_vartype
def test_jit_all_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.all(inputs)
    inputs=np.array([True, False, True])
    dtype=["bool]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.all
        """
        a = paddle.all(inputs)
        return a

    inps = np.array([True, False, True])
    runner = Runner(func=func, name="all_1", dtype=["bool"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_all_parameters
def test_jit_all_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.all(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.all
        """
        return paddle.all(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2]).astype(np.bool)
    runner = Runner(func=func, name="all_2", dtype=["bool"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
