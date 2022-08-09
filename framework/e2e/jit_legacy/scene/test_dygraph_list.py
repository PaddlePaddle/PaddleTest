#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test dygraph list
"""
import pytest
import paddle
import numpy as np


@pytest.mark.jit_dygraph_list_vartype
def test_jit_dygraph_list_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.abs(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func():
        x = paddle.rand([1, 2])
        y = paddle.rand([2, 2])
        bs = paddle.shape(x)[0]
        # it will be converted into tensor_array
        arr = []
        for i in bs:
            arr.append(x)
            arr.append(y)
        shape = [x.shape for x in arr]
        return shape

    res = func()
    exp = [(2, 2), (2, 2)]
    assert np.allclose(res, exp)
