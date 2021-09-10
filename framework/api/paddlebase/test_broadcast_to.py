#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_broadcast_to.py
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestBroadcast(APIBase):
    """
    test broadcast
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestBroadcast(paddle.broadcast_to)


@pytest.mark.api_base_broadcast_to_vartype
def test_broadcast_to_base():
    """
    broadcast_to_base
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    out_shape = [2, 6]
    res = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    obj.base(res=res, x=x_data, shape=out_shape)


@pytest.mark.api_base_broadcast_to_prarmeters
def test_broadcast_to():
    """
    broadcast_to: 1D to 2D on dim0, float32
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    out_shape = [2, 6]
    res = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    obj.run(res=res, x=x_data, shape=out_shape)


@pytest.mark.api_base_broadcast_to_prarmeters
def test_broadcast_to1():
    """
    broadcast_to1: 1D to 2D on dim1, float64
    """
    x_data = np.arange(1, 7).reshape((6, 1)).astype(np.float64)
    out_shape = [6, 2]
    res = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
    obj.run(res=res, x=x_data, shape=out_shape)


@pytest.mark.api_base_broadcast_to_prarmeters
def test_broadcast_to2():
    """
    broadcast_to2: 1D to 2D on dim0, int32
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.int32)
    out_shape = [2, 6]
    res = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    obj.run(res=res, x=x_data, shape=out_shape)


@pytest.mark.api_base_broadcast_to_prarmeters
def test_broadcast_to3():
    """
    broadcast_to3: 1D to 2D on dim1, int64
    """
    x_data = np.arange(1, 7).reshape((6, 1)).astype(np.int64)
    out_shape = [6, 2]
    res = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    obj.run(res=res, x=x_data, shape=out_shape)


@pytest.mark.api_base_broadcast_to_prarmeters
def test_broadcast_to4():
    """
    broadcast_to4: 3D to 4D on dim0, float32
    """
    x_data = np.arange(1, 19).reshape((2, 3, 3)).astype(np.float32)
    out_shape = [3, 2, 3, 3]
    res = np.array(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ],
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ],
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ],
        ]
    )

    obj.run(res=res, x=x_data, shape=out_shape)


# def test_broadcast_to5():
#     """
#     broadcast_to5: 1D to 6D on dim0, float32
#     """
#     x_data = np.arange(1, 3).reshape((2, 1)).astype(np.float32)
#     out_shape = [3, 2, 2, 2, 3, 2]
#     res = np.array([[[[[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]],
#
#
#           [[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]]],
#
#
#
#          [[[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]],
#
#
#           [[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]]]],
#
#
#
#
#         [[[[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]],
#
#
#           [[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]]],
#
#
#
#          [[[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]],
#
#
#           [[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]]]],
#
#
#
#
#         [[[[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]],
#
#
#           [[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]]],
#
#
#
#          [[[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]],
#
#
#           [[[1., 2.],
#             [1., 2.],
#             [1., 2.]],
#
#            [[1., 2.],
#             [1., 2.],
#             [1., 2.]]]]]])
#
#     obj.run(res=res, x=x_data, shape=out_shape)


@pytest.mark.api_base_broadcast_to_prarmeters
def test_broadcast_to6():
    """
    broadcast_to6: 1D to 2D on dim0, bool
    """
    x_data = np.array([0, 1]).reshape((2,)).astype(np.bool)
    out_shape = [5, 2]
    res = np.array([[False, True], [False, True], [False, True], [False, True], [False, True]])

    obj.run(res=res, x=x_data, shape=out_shape)
