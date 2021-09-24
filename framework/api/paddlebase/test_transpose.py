#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_transpose
"""
from apibase import APIBase
from apibase import compare
import paddle
import pytest
import numpy as np

# import math


class TestTranspose(APIBase):
    """
    test transpose
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestTranspose(paddle.transpose)


@pytest.mark.api_base_transpose_vartype
def test_transpose_base():
    """
    x.shape=(1, 2, 2, 3)
    res.shape=(1, 3, 2, 2)
    float32
    """
    x = np.ones([1, 2, 2, 3]).astype(np.float32)
    perm = [0, 3, 1, 2]
    res = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]]).astype(np.float32)
    obj.base(res=res, x=x, perm=perm)


@pytest.mark.api_base_transpose_parameters
def test_transpose1():
    """
    x.shape=(1, 2, 2, 3)
    res.shape=(1, 3, 2, 2)
    int32
    """
    x = np.ones([1, 2, 2, 3]).astype(np.int32)
    perm = [0, 3, 1, 2]
    res = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]]).astype(np.int32)
    obj.run(res=res, x=x, perm=perm)


@pytest.mark.api_base_transpose_parameters
def test_transpose2():
    """
    x.shape=(3, 1, 2, 2)
    res.shape=(1, 3, 2, 2)
    float64
    """
    x = np.arange(1, 13).reshape((3, 1, 2, 2)).astype(np.float64)
    perm = [1, 0, 2, 3]
    res = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]]).astype(
        np.float64
    )
    obj.run(res=res, x=x, perm=perm)


@pytest.mark.api_base_transpose_parameters
def test_transpose3():
    """
    x.shape=(1, 3, 2, 2)
    res.shape=(3, 1, 2, 2)
    bool
    """
    paddle.disable_static()
    x = np.array(
        [[[[False, True], [True, False]], [[False, False], [True, False]], [[True, True], [False, True]]]]
    ).astype(np.bool)
    exp = np.array(
        [[[[False, True], [True, False]]], [[[False, False], [True, False]]], [[[True, True], [False, True]]]]
    ).astype(np.bool)
    x_p = paddle.to_tensor(x)
    perm = [1, 0, 2, 3]
    res = paddle.transpose(x_p, perm)
    compare(res.numpy(), exp)


@pytest.mark.api_base_transpose_parameters
def test_transpose4():
    """
    x.shape=(2, 1, 3, 4, 2, 1, 2)
    res.shape=(2, 2, 1, 2, 4, 1, 3)
    bool
    """
    paddle.disable_static()
    x = np.random.random((2, 1, 3, 4, 2, 1, 2)) * 4 - 2
    x_p = paddle.to_tensor(x.astype(np.bool))
    perm = [6, 0, 1, 4, 3, 5, 2]
    exp = np.transpose(x.astype(np.bool), perm)
    res = paddle.transpose(x_p, perm)
    compare(res.numpy(), exp)
