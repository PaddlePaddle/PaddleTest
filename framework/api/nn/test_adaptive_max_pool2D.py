#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_adaptive_max_pool2D
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAdaptiveMaxPool2D(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestAdaptiveMaxPool2D(paddle.nn.AdaptiveMaxPool2D)


@pytest.mark.api_nn_AdaptiveAvgPool2D_vartype
def test_adaptive_max_pool2D_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 1, 8, 8])
    output_size = 3
    res = np.array(
        [
            [
                [
                    [9.30053640, 9.30053640, 9.66333642],
                    [9.06944590, 9.06944590, 9.40579439],
                    [4.20803488, 6.04654429, 5.46431549],
                ]
            ],
            [
                [
                    [9.43649840, 7.96382733, 6.68005243],
                    [9.45728423, 9.45728423, 9.27742528],
                    [9.87437073, 9.73072488, 8.73190462],
                ]
            ],
        ]
    )
    obj.base(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_max_pool2D1():
    """
    default
    """
    x = randtool("float", -10, 10, [1, 2, 8, 8])
    output_size = 4
    res = np.array(
        [
            [
                [
                    [7.15934614, 5.34758122, 4.4535757, 4.37578918],
                    [7.4850067, 6.05787992, 6.13915722, 4.46948541],
                    [4.24844101, 8.73619308, 6.0619232, 1.35578121],
                    [7.12392511, 2.97429179, 5.8461423, 5.62927583],
                ],
                [
                    [8.78091997, 9.0289969, 8.00528653, 8.75545006],
                    [3.76045629, 8.69053661, 7.14328067, 8.31464399],
                    [1.35203085, 6.7995525, 6.7986085, 9.51210438],
                    [8.36070629, 7.81533464, 9.38994375, 9.68484741],
                ],
            ]
        ]
    )
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_max_pool2D2():
    """
    exception return_mask=True
    """
    x = randtool("float", -10, 10, [1, 1, 5, 5])
    output_size = 3
    res = np.array(
        [
            [[[[9.721807, 8.592546, 8.225645], [5.9005647, 9.36555, 9.36555], [9.518723, 9.518723, 9.36555]]]],
            [[[[0, 1, 9], [15, 18, 18], [21, 21, 18]]]],
        ]
    )
    obj.static = False
    obj.base(res=res, data=x, output_size=output_size, return_mask=True)
    obj.static = True
