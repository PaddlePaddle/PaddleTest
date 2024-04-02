#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test empty
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_empty_vartype
def test_jit_empty_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.empty(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.empty
        """
        return paddle.empty(inputs)

    inps = np.array([3, 2, 5])
    runner = Runner(func=func, name="empty_base", dtype=["int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
