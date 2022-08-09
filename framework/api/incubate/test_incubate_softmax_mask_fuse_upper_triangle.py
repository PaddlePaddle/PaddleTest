#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.incubate.softmax_mask_fuse_upper_triangle
"""
import sys
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np

sys.path.append("../..")
from utils.interceptor import skip_not_compile_gpu


class TestIncubateSoftmaxMaskFuseUpperTriangle(APIBase):
    """
    test abs
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestIncubateSoftmaxMaskFuseUpperTriangle(paddle.incubate.softmax_mask_fuse_upper_triangle)
obj1 = TestIncubateSoftmaxMaskFuseUpperTriangle(paddle.incubate.softmax_mask_fuse_upper_triangle)


def get_softmax_upper(x, dtype="float32"):
    """
    naive softmax_mask_fuse_upper_triangle
    """
    x_lower = np.tril(x)
    masked_x = np.where(x_lower == 0, -10000.0, x_lower).astype("float32")
    max_value = np.max(masked_x, axis=-1, keepdims=True)
    before_exp = masked_x - max_value
    exp = np.exp(before_exp)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    rst = exp / exp_sum
    rst = rst.astype(dtype)
    return rst


@skip_not_compile_gpu
@pytest.mark.api_base_abs_vartype
def test_incubate_softmax_mask_fuse_upper_triangle_base():
    """
    base
    np.float32
    x.shape=[1, 1, 32, 32]
    """
    obj.places = [paddle.CUDAPlace(0)]
    x = randtool("float", -1, 1, (1, 1, 32, 32))
    res = get_softmax_upper(x, dtype="float32")
    obj.base(res=res, x=x)


@skip_not_compile_gpu
@pytest.mark.api_base_abs_vartype
def test_incubate_softmax_mask_fuse_upper_triangle_base1():
    """
    base
    np.float16
    x.shape=[1, 1, 32, 32]
    """
    obj1.places = [paddle.CUDAPlace(0)]
    obj1.types = [np.float16]
    x = randtool("float", -1, 1, (1, 1, 32, 32)).astype(np.float16)
    res = get_softmax_upper(x, dtype="float16")
    obj1.base(res=res, x=x)


@skip_not_compile_gpu
@pytest.mark.api_base_abs_parameters
def test_incubate_softmax_mask_fuse_upper_triangle_1():
    """
    np.float32
    x.shape=[3, 1, 32, 32]
    """
    obj.places = [paddle.CUDAPlace(0)]
    x = randtool("float", -1, 1, (3, 1, 224, 224))
    res = get_softmax_upper(x, dtype="float32")
    obj.run(res=res, x=x)


@skip_not_compile_gpu
@pytest.mark.api_base_abs_parameters
def test_incubate_softmax_mask_fuse_upper_triangle_2():
    """
    base
    """
    obj.places = [paddle.CUDAPlace(0)]
    x = randtool("float", -1, 1, (5, 7, 224, 224))
    res = get_softmax_upper(x, dtype="float32")
    obj.run(res=res, x=x)


@skip_not_compile_gpu
@pytest.mark.api_base_abs_parameters
def test_incubate_softmax_mask_fuse_upper_triangle_3():
    """
    base
    """
    obj.places = [paddle.CUDAPlace(0)]
    x = randtool("float", -1, 1, (7, 11, 32, 32))
    res = get_softmax_upper(x, dtype="float32")
    obj.run(res=res, x=x)
