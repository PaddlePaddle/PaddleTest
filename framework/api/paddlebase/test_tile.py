#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_tile.py
"""

from apibase import APIBase
from apibase import randtool
from apibase import compare

import paddle
import pytest
import numpy as np


class TestTile(APIBase):
    """
    test tile
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.no_grad_var = {"x", "repeat_times"}
        # self.enable_backward = False


obj = TestTile(paddle.tile)


@pytest.mark.api_base_tile_vartype
def test_tile_base():
    """
    base,shape=list
    """
    x = randtool("float", -10, 10, [2, 3])
    repeat_times = [3, 2]
    res = np.tile(x, repeat_times)
    obj.base(res=res, x=x, repeat_times=repeat_times)


@pytest.mark.api_base_tile_parameters
def test_tile():
    """
    repeat_times=tuple
    """
    x = randtool("float", -10, 10, [1, 3, 2])
    repeat_times = (2, 3)
    res = np.tile(x, repeat_times)
    obj.run(res=res, x=x, repeat_times=repeat_times)


obj.enable_backward = False


# def test_tile1():
#     """
#     x_type=bool,shape=list,ok,case failed, bool not support backword, icafe
#     """
#     x = np.array([False, True])
#     repeat_times = [1, 3, 2]
#     res = np.tile(x, repeat_times)
#     obj.run(res=res, x=x, repeat_times=repeat_times)


@pytest.mark.api_base_tile_parameters
def test_tile2():
    """
    repeat_times = (3,),x_type=np.int32
    """
    x = np.array([1, 2, 3]).astype(np.int32)
    repeat_times = (3,)
    res = np.tile(x, 3)
    obj.run(res=res, x=x, repeat_times=repeat_times)


@pytest.mark.api_base_tile_parameters
def test_tile3():
    """
    repeat_times_type=np.int32
    """
    x = np.array([1, 2, 3])
    repeat_times = np.array([2, 3]).astype(np.int32)
    res = np.tile(x, [2, 3])
    obj.run(res=res, x=x, repeat_times=repeat_times)


@pytest.mark.api_base_tile_parameters
def test_tile4():
    """
    repeat_times = 'int32'
    """
    x = np.array([1, 2, 3])
    repeat_times = np.array([3, 2]).astype("int32")
    res = np.tile(x, [3, 2])
    obj.run(res=res, x=x, repeat_times=repeat_times)


@pytest.mark.api_base_tile_parameters
def test_tile5():
    """
    x_type=np.int64
    """
    x = np.array([1, 2, 3]).astype(np.int64)
    repeat_times = np.array([6, 4]).astype(np.int32)
    res = np.tile(x, [6, 4])
    obj.run(res=res, x=x, repeat_times=repeat_times)


@pytest.mark.api_base_tile_parameters
def test_tile6():
    """
    shape = [1, 3, 4, 4, 1, 1]
    """
    paddle.disable_static()
    x = paddle.ones([1, 3, 1, 1, 1, 1])
    x.stop_gradient = False
    y = paddle.tile(x, [1, 3, 4, 4, 1, 1])
    y = y.sum()
    y.backward(retain_graph=True)
    res = x.grad
    exp = np.array([[[[[[48.0]]]], [[[[48.0]]]], [[[[48.0]]]]]])
    compare(res.numpy(), exp)
