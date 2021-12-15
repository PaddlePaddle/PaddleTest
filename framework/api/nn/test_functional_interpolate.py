#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_interpolate
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestInterpolate(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestInterpolate(paddle.nn.functional.interpolate)


@pytest.mark.api_nn_interpolate_vartype
def test_instancenorm_base():
    """
    size=[6, 6]
    input.dtype: float32
    """
    x_data = np.array(
        [[[[1, 0, 1, 30], [3, 2, 0, 22], [-1, 1, 2, -11]]], [[[-3, 2, 5, -23], [-2, 3, 21, -12], [1, -2, 20, 33]]]]
    ).astype(np.float32)

    res = [
        [
            [
                [1.0, 1.0, 0.0, 1.0, 1.0, 30.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 30.0],
                [3.0, 3.0, 2.0, 0.0, 0.0, 22.0],
                [3.0, 3.0, 2.0, 0.0, 0.0, 22.0],
                [-1.0, -1.0, 1.0, 2.0, 2.0, -11.0],
                [-1.0, -1.0, 1.0, 2.0, 2.0, -11.0],
            ]
        ],
        [
            [
                [-3.0, -3.0, 2.0, 5.0, 5.0, -23.0],
                [-3.0, -3.0, 2.0, 5.0, 5.0, -23.0],
                [-2.0, -2.0, 3.0, 21.0, 21.0, -12.0],
                [-2.0, -2.0, 3.0, 21.0, 21.0, -12.0],
                [1.0, 1.0, -2.0, 20.0, 20.0, 33.0],
                [1.0, 1.0, -2.0, 20.0, 20.0, 33.0],
            ]
        ],
    ]

    size = [6, 6]
    obj.base(res=res, x=x_data, size=size)


@pytest.mark.api_nn_interpolate_parameters
def test_instancenorm1():
    """
    size=[6, 6]
    input.dtype: int32
    """
    x_data = np.array(
        [[[[1, 0, 1, 30], [3, 2, 0, 22], [-1, 1, 2, -11]]], [[[-3, 2, 5, -23], [-2, 3, 21, -12], [1, -2, 20, 33]]]]
    ).astype(np.int32)

    res = [
        [
            [
                [1, 1, 0, 1, 1, 30],
                [1, 1, 0, 1, 1, 30],
                [3, 3, 2, 0, 0, 22],
                [3, 3, 2, 0, 0, 22],
                [-1, -1, 1, 2, 2, -11],
                [-1, -1, 1, 2, 2, -11],
            ]
        ],
        [
            [
                [-3, -3, 2, 5, 5, -23],
                [-3, -3, 2, 5, 5, -23],
                [-2, -2, 3, 21, 21, -12],
                [-2, -2, 3, 21, 21, -12],
                [1, 1, -2, 20, 20, 33],
                [1, 1, -2, 20, 20, 33],
            ]
        ],
    ]

    size = [6, 6]
    obj.run(res=res, x=x_data, size=size)


@pytest.mark.api_nn_interpolate_parameters
def test_instancenorm2():
    """
    scale_factor = [2, 1.5]
    input.dtype: int32
    """
    x_data = np.array(
        [[[[1, 0, 1, 30], [3, 2, 0, 22], [-1, 1, 2, -11]]], [[[-3, 2, 5, -23], [-2, 3, 21, -12], [1, -2, 20, 33]]]]
    ).astype(np.int64)

    res = [
        [
            [
                [1, 1, 0, 1, 1, 30],
                [1, 1, 0, 1, 1, 30],
                [3, 3, 2, 0, 0, 22],
                [3, 3, 2, 0, 0, 22],
                [-1, -1, 1, 2, 2, -11],
                [-1, -1, 1, 2, 2, -11],
            ]
        ],
        [
            [
                [-3, -3, 2, 5, 5, -23],
                [-3, -3, 2, 5, 5, -23],
                [-2, -2, 3, 21, 21, -12],
                [-2, -2, 3, 21, 21, -12],
                [1, 1, -2, 20, 20, 33],
                [1, 1, -2, 20, 20, 33],
            ]
        ],
    ]

    scale_factor = [2, 1.5]
    obj.run(res=res, x=x_data, scale_factor=scale_factor)


# def test_instancenorm3():
#     """
#     scale_factor = [2, 1.5]
#     input.dtype: float64
#     mode = "bilinear"
#     """
#     x_data = np.array([[[[1., 0., 1., 30.],
#                          [3., 2., 0., 22.],
#                          [-1., 1., 2., -11.]]],
#                        [[[-3., 2., 5., -23.],
#                          [-2., 3., 21., -12.],
#                          [1., -2., 20., 33.]]]]).astype(np.float64)
#
#     res = [[[[1., 0.5, 0.16666675, 0.83333337, 15.5, 30.],
#              [1.5, 1., 0.54166669, 0.70833334, 14.375, 28.],
#              [2.5, 2., 1.29166657, 0.45833328, 12.125, 24.],
#              [2., 1.875, 1.54166657, 0.70833328, 7.125, 13.75],
#              [0., 0.62500000, 1.29166669, 1.45833334, -0.625, -2.75],
#              [-1., 0., 1.16666675, 1.83333337, -4.5, -11.]]],
#             [[-3., -0.50, 2.50000024, 4.50000012, -9., -23.],
#              [-2.7, -0.25, 3.37500054, 7.87500027, -5.625, -20.25],
#              [-2.2, 0.25, 5.12500113, 14.62500057, 1.125, -14.75],
#              [-1.2, 0.25, 4.91666818, 17.58333409, 10., -0.750],
#              [0.2, -0.25, 2.75000167, 16.75000083, 21., 21.75],
#              [1., -0.50, 1.66666842, 16.33333421, 26.50, 33.]]]
#
#     scale_factor = [2, 1.5]
#     mode = "bilinear"
#     obj.base(res=res, x=x_data, scale_factor=scale_factor, mode=mode)
