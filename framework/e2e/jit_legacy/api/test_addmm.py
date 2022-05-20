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


@pytest.mark.jit_addmm_vartype
def test_jit_addmm_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.addmm(inputs, inputs1, inputs2)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs1, inputs2):
        """
        paddle.addmm
        """
        return paddle.addmm(inputs, inputs1, inputs2)

    inps = randtool("float", -2, 2, shape=[3, 3])
    runner = Runner(func=func, name="add_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs1=inps, inputs2=inps)
    runner.run()


@pytest.mark.jit_addmm_vartype
def test_jit_addmm_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.addmm(inputs, inputs1, inputs2)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs1, inputs2):
        """
        paddle.addmm
        """
        a = paddle.addmm(inputs, inputs1, inputs2)
        return a

    inps = randtool("float", -2, 2, shape=[3, 3])
    runner = Runner(func=func, name="add_1", dtype=["float32", "float64"], ftype="func")
    runner.delta = 1e-6
    runner.rtol = 1e-7
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs1=inps, inputs2=inps)
    runner.run()
