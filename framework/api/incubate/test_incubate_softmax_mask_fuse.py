#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.incubate.softmax_mask_fuse
"""
import os
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestIncubateSoftmaxMaskFuse(APIBase):
    """
    test abs
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestIncubateSoftmaxMaskFuse(paddle.incubate.softmax_mask_fuse)
obj.places = [paddle.CUDAPlace(0)]


def get_softmax(x, mask, dtype="float16"):
    """
    naive softmax_mask_fuse
    """
    masked_x = (x + mask).astype("float32")
    max_value = np.max(masked_x, axis=-1, keepdims=True)
    before_exp = masked_x - max_value
    exp = np.exp(before_exp)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    rst = exp / exp_sum
    rst = rst.astype(dtype)
    return rst


@pytest.mark.api_base_abs_vartype
def test_incubate_softmax_mask_fuse_base():
    """
    base
    np.float16
    x.shape=(2, 8, 8, 32)
    """
    x = randtool("float", -1, 1, (2, 8, 8, 32)).astype("float16")
    mask = np.random.randint(0, 2, (2, 1, 8, 32)).astype("float16")
    mask = np.where(mask == 1, -10000.0, mask)
    res = get_softmax(x, mask, dtype="float16")
    obj.base(res=res, x=x, mask=mask)


@pytest.mark.api_base_abs_parameters
def test_incubate_softmax_mask_fuse_1():
    """
    np.float16
    x.shape=(2, 8, 8, 1020)
    """
    x = randtool("float", -1, 1, (2, 8, 8, 1020)).astype("float16")
    mask = np.random.randint(0, 2, (2, 1, 8, 1020)).astype("float16")
    mask = np.where(mask == 1, -10000.0, mask)
    res = get_softmax(x, mask, dtype="float16")
    obj.delta = 1e-5
    obj.rtol = 1e-6
    obj.run(res=res, x=x, mask=mask)
    obj.delta = 1e-6
    obj.rtol = 1e-7


@pytest.mark.api_base_abs_parameters
def test_incubate_softmax_mask_fuse_2():
    """
    np.float16
    x.shape=(6, 8, 8, 32)
    """
    x = randtool("float", -1, 1, (6, 8, 8, 32)).astype("float16")
    mask = np.random.randint(0, 2, (6, 1, 8, 32)).astype("float16")
    mask = np.where(mask == 1, -10000.0, mask)
    res = get_softmax(x, mask, dtype="float16")
    obj.run(res=res, x=x, mask=mask)


@pytest.mark.api_base_abs_parameters
def test_incubate_softmax_mask_fuse_3():
    """
    np.float16
    x.shape=(7, 3, 5, 32)
    """
    x = randtool("float", -1, 1, (7, 3, 16, 32)).astype("float16")
    mask = np.random.randint(0, 2, (7, 1, 16, 32)).astype("float16")
    mask = np.where(mask == 1, -10000.0, mask)
    res = get_softmax(x, mask, dtype="float16")
    obj.run(res=res, x=x, mask=mask)
