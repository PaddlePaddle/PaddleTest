#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test add
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_allclose_vartype
def test_jit_allclose_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.allclose(inputs, inputs1)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs1):
        """
        paddle.allclose
        """
        return paddle.allclose(inputs, inputs1)

    inps = randtool("float", -2, 2, shape=[3, 3])
    runner = Runner(func=func, name="add_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs1=inps)
    runner.run()


@pytest.mark.jit_allclose_vartype
def test_jit_allclose_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.allclose(inputs, inputs1)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs1):
        """
        paddle.allclose
        """
        a = paddle.allclose(inputs, inputs1)
        return a

    inps = randtool("float", -2, 2, shape=[3, 3])
    runner = Runner(func=func, name="add_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs1=inps)
    runner.run()
