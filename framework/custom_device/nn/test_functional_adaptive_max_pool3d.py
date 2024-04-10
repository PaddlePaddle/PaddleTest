#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_adaptive_max_pool3D
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAdaptiveMaxPool3D(APIBase):
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


obj = TestAdaptiveMaxPool3D(paddle.nn.functional.adaptive_max_pool3d)
obj1 = TestAdaptiveMaxPool3D(paddle.nn.functional.adaptive_max_pool3d)


@pytest.mark.api_nn_AdaptiveAvgPool3D_vartype
def test_adaptive_max_pool3D_base():
    """
    default
    """
    x = randtool("float", -10, 10, [1, 2, 8, 8, 8])
    output_size = 4
    res = np.array(
        [
            [
                [
                    [
                        [9.436499, 9.300536, 7.4079137, 9.663337],
                        [9.352632, 9.457284, 4.459953, 9.405794],
                        [9.35952, 7.0134306, 6.1536007, 6.8121696],
                        [9.874371, 9.730725, 8.731905, 4.0392017],
                    ],
                    [
                        [8.78092, 9.028996, 8.005286, 8.75545],
                        [7.485007, 8.6905365, 7.1432805, 8.314644],
                        [4.248441, 8.736193, 6.7986083, 9.512104],
                        [8.360706, 7.815335, 9.389944, 9.684848],
                    ],
                    [
                        [9.721807, 7.606751, 7.3338304, 6.744018],
                        [8.141745, 9.36555, 9.518723, 9.274939],
                        [7.0962505, 7.3109465, 7.948995, 8.901449],
                        [9.581015, 8.288019, 8.45119, 9.769033],
                    ],
                    [
                        [4.4988766, 8.364453, 9.335013, 5.4339304],
                        [8.279349, 7.240312, 9.881113, 9.404358],
                        [8.090834, 7.1410418, 9.64819, 5.7836027],
                        [9.916014, 8.718832, 9.619197, 9.778963],
                    ],
                ],
                [
                    [
                        [8.241472, 8.387796, 8.862762, 9.965746],
                        [9.450807, 9.8903265, 8.301665, 8.609721],
                        [8.153333, 2.1524062, 0.89549863, 8.196441],
                        [9.83181, 9.813317, 9.475018, 9.168265],
                    ],
                    [
                        [8.874169, 6.8945503, 9.974066, 8.032424],
                        [5.6218143, 9.832883, 9.848681, 9.158935],
                        [7.862855, 8.494994, 9.513051, 8.418136],
                        [7.47018, 9.104637, 7.8543377, 7.573076],
                    ],
                    [
                        [4.3139825, 9.743325, 9.536787, 6.2394576],
                        [6.448162, 9.262486, 7.8650265, 9.737822],
                        [8.92775, 7.9192333, 2.3204846, 6.8921795],
                        [9.758237, 2.717995, 9.532545, 7.116392],
                    ],
                    [
                        [8.165372, 8.017488, 9.893595, 8.808366],
                        [7.6648765, 9.003139, 9.420015, 9.261234],
                        [9.035168, 9.255864, 8.724097, 8.944263],
                        [9.286772, 6.3624268, 6.74282, 8.614413],
                    ],
                ],
            ]
        ]
    )
    obj.base(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_max_pool3D2():
    """
    exception return_mask=True
    """
    x = randtool("float", -10, 10, [1, 1, 5, 5, 5])
    output_size = 3
    res = np.array(
        [
            [
                [
                    [
                        [
                            [7.582787, 9.857272, 2.265059],
                            [5.4414554, 9.857272, 9.992919],
                            [8.154518, 5.746608, 9.992919],
                        ],
                        [
                            [7.203608, 8.008818, 9.025089],
                            [6.0899134, 9.595793, 9.992919],
                            [9.400332, 7.356853, 9.992919],
                        ],
                        [
                            [4.9811325, 8.157785, 8.157785],
                            [6.0899134, 8.592143, 8.592143],
                            [9.400332, 5.377379, 5.377379],
                        ],
                    ]
                ]
            ],
            [
                [
                    [
                        [[1, 7, 9], [10, 7, 44], [45, 42, 44]],
                        [[51, 32, 54], [85, 63, 44], [95, 67, 44]],
                        [[101, 108, 108], [85, 113, 113], [95, 118, 118]],
                    ]
                ]
            ],
        ]
    )
    obj1.static = False
    obj1.base(res=res, x=x, output_size=output_size, return_mask=True)
