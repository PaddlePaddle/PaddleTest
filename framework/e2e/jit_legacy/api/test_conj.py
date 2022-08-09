#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test conj
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_conj_vartype
def test_jit_conj_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.conj(inputs)
    inputs=np.array([[1.5, 2.1, 3.2], [4.1, 3.2, 2.2]])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.conj
        """
        return paddle.conj(inputs)

    inps = np.array([[1.5, 2.1, 3.2], [4.1, 3.2, 2.2]])
    runner = Runner(func=func, name="conj_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_conj_parameters
def test_jit_conj_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.conj(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.conj
        """
        return paddle.conj(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="conj_2", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_conj_parameters
def test_jit_conj_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.conj(inputs)
    inputs=np.array([[1.5+1j, 2.1+2j, 3.2+3j], [4.1+4j, 3.2+5j, 2.2+6j]])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.conj
        """
        return paddle.conj(inputs)

    inps = np.array([[1.5 + 1j, 2.1 + 2j, 3.2 + 3j], [4.1 + 4j, 3.2 + 5j, 2.2 + 6j]])
    runner = Runner(func=func, name="conj_2", dtype=["complex64", "complex128"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
